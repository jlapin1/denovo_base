from __future__ import annotations

import functools
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from models.insertdelete import custom_schedules
from models.insertdelete.torch_forward_process import (
    ReverseProcessMarginalDistributionTorch,
    safe_exp_weighted_torch,
)

try:
    import jax
    import jax.numpy as jnp
    from google_d3pm_ins_del import forward_process as gfp
    from google_d3pm_ins_del import util as gutil
    from models.insertdelete.bridge import InsertDeleteTorchJaxBridge
    from models.insertdelete.parameterizations import InsertDeleteX0Parameterizer
except Exception as exc:  # pragma: no cover - explicit runtime guard.
    jax = None
    jnp = None
    gfp = None
    gutil = None
    InsertDeleteTorchJaxBridge = None
    InsertDeleteX0Parameterizer = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


class InsertDeleteDiffusion:
    """Insert/delete diffusion backend using vendored Google forward-process math.

    Design notes:
    - Forward noising/alignment/posterior targets are sourced from
      `google_d3pm_ins_del/*` modules.
    - Model optimization stays in Torch (decoder + extra heads), with targets
      imported from the JAX forward-process routines.
    """

    def __init__(self, config: Dict, dictionary: Dict, backbone) -> None:
        """Initialize insert/delete diffusion backend.

        Args:
            config: Insert/delete diffusion config subtree.
            dictionary: Token->id mapping used by the Torch decoder.
            backbone: Torch decoder module with insert/delete heads.
        """
        if jax is None or jnp is None:
            raise ImportError(
                "InsertDeleteDiffusion requires JAX. "
                f"Original import error: {_JAX_IMPORT_ERROR}"
            )

        self.config = config
        self.backbone = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_steps = int(config["T"])
        self.time_conditioning = bool(config.get("time_conditioning", True))
        self.self_condition = bool(config.get("model", {}).get("self_condition", True))
        self.model_prediction = str(config.get("model_prediction", "x0")).lower()
        if self.model_prediction == "x_start":
            self.model_prediction = "x0"
        if self.model_prediction not in {"x0", "direct"}:
            raise NotImplementedError(
                "InsertDeleteDiffusion supports model_prediction in {'x0', 'direct'}."
            )
        self.max_len = int(config["model"]["length"])

        schedule_cfg = config.get("schedule", {})
        transition_type = str(schedule_cfg.get("transition_type", "uniform")).lower()
        insert_type = str(schedule_cfg.get("insert_type", "uniform")).lower()
        include_mask_in_vocab = (
            transition_type in {"mask", "delayed_mask"} or insert_type == "mask"
        )

        self.bridge = InsertDeleteTorchJaxBridge(
            dictionary,
            max_len=self.max_len,
            include_mask_in_vocab=include_mask_in_vocab,
        )
        self.pad_token_id = self.bridge.pad_token_id
        self.eos_token_id = self.bridge.eos_token_id
        self.sos_token_id = self.bridge.sos_token_id
        self.mask_token_id = self.bridge.mask_token_id
        self.insert_token_id = self.bridge.insert_token_id
        self.delete_token_id = self.bridge.delete_token_id
        self.real_torch_ids_tensor = self.bridge.real_torch_ids_tensor

        deny_insert_jax = self.bridge.resolve_deny_insert_jax(
            schedule_cfg=schedule_cfg, dictionary=dictionary
        )
        mask_token_jax = self.bridge.mask_token_jax()

        # ref repo google_d3pm_ins_del/training_setup.py builds schedules from
        # interpolators with transition/insert/refresh controls; delegate this
        # logic to local custom_schedules (wrapper-only, no vendored edits).
        self.schedule = custom_schedules.build_schedule_from_config(
            num_steps=self.num_steps,
            vocab_size=self.bridge.real_vocab_size,
            schedule_cfg=schedule_cfg,
            token_denylist=deny_insert_jax,
            mask_token=mask_token_jax,
            max_len=self.max_len,
        )

        self.loss_weight_token = float(config.get("loss_weights", {}).get("token", 1.0))
        self.loss_weight_insert = float(
            config.get("loss_weights", {}).get("insert", 1.0)
        )
        self.loss_weight_delete = float(
            config.get("loss_weights", {}).get("delete", 1.0)
        )
        self.loss_weight_length = float(
            config.get("loss_weights", {}).get("length", 1.0)
        )

        self.max_rejects = int(schedule_cfg.get("max_rejects", 16))
        self.sampling_steps = int(
            config.get("sampling", {}).get("steps", self.num_steps)
        )
        self.sampling_top = int(config.get("sampling", {}).get("top", 1))
        if self.sampling_steps != self.num_steps:
            raise ValueError(
                "Insert/delete requires sampling.steps == T for a mathematically "
                "faithful reverse chain."
            )

        seed = int(config.get("seed", 0))
        self.jax_key = jax.random.PRNGKey(seed)
        self._build_jax_target_fns()
        self._build_jax_sampling_fns()
        self.x0_parameterizer = None
        if self.model_prediction == "x0":
            self.x0_parameterizer = InsertDeleteX0Parameterizer(self.schedule)

    def _split_key(self, num: int = 1):
        """Split and update RNG state.

        Args:
            num: Number of fresh keys to return.

        Returns:
            Single key if `num == 1`, otherwise an array of keys.
        """
        keys = jax.random.split(self.jax_key, num + 1)
        self.jax_key = keys[0]
        if num == 1:
            return keys[1]
        return keys[1:]

    def _build_jax_target_fns(self) -> None:
        """Build JIT-compiled JAX functions for sampling training targets."""
        schedule = self.schedule
        max_rejects = self.max_rejects
        terminal_distn = schedule.terminal_distn()

        def _single_target(
            x0_seq: gfp.DynamicLengthSentinelSequence, rng: jax.Array
        ) -> Dict[str, jax.Array]:
            t_key, sample_key = jax.random.split(rng)
            t = schedule.sample_step_number(t_key)
            t_weight = schedule.weights[t]

            d_0_to_t, d_t_to_tplus1 = schedule.distns_at_step(t)
            d_0_to_tplus1, _ = schedule.distns_at_step(t + 1)
            precomputed_x0_A0tplus1 = jax.vmap(
                functools.partial(d_0_to_tplus1.A.apply, is_distn=False, log=True)
            )(x0_seq.tokens)

            def _generate_sample(key):
                xtplus1, align_0_tplus1 = gfp.sample_noise_step(
                    x0_seq,
                    d_0_to_tplus1,
                    key,
                    precomputed_xA=precomputed_x0_A0tplus1,
                )
                is_valid = xtplus1.is_valid()
                return is_valid, (is_valid, xtplus1, align_0_tplus1)

            is_valid, xtplus1, align_0_tplus1 = gutil.rejection_sample(
                _generate_sample,
                sample_key,
                max_rejects=max_rejects,
            )

            precomputed_x0_A0t = jax.vmap(
                functools.partial(d_0_to_t.A.apply, is_distn=False, log=True)
            )(x0_seq.tokens)
            precomputed_Attplus1_xtplus1 = jax.vmap(
                functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True)
            )(xtplus1.tokens)
            denoise_targets, expected_q_xtplus1_given_xt = gfp.intermediate_marginals(
                x0=x0_seq,
                xtplus1=xtplus1,
                alignment=align_0_tplus1,
                d_0_to_t=d_0_to_t,
                d_t_to_tplus1=d_t_to_tplus1,
                precomputed_x0_A0t=precomputed_x0_A0t,
                precomputed_Attplus1_xtplus1=precomputed_Attplus1_xtplus1,
            )
            xT_length_dist = gfp.terminal_sentinel_distribution(x0_seq, terminal_distn)
            return {
                "t": t,
                "t_weight": t_weight,
                "xt_tokens": xtplus1.tokens,
                "xt_length": xtplus1.length,
                "expected_q": expected_q_xtplus1_given_xt,
                "valid": is_valid.astype(jnp.float32),
                "target_was_missing": denoise_targets.was_missing,
                "target_was_insert_logprob": denoise_targets.was_insert_log_prob,
                "target_prev_token_logprob": denoise_targets.previous_token_log_probs,
                "target_delete_logprob": denoise_targets.log_prob_number_of_preceding_deletes,
                "target_length": denoise_targets.length,
                "target_length_logprob": xT_length_dist,
            }

        self._jax_batched_targets = jax.jit(jax.vmap(_single_target, in_axes=(0, 0)))

    def _build_jax_sampling_fns(self) -> None:
        """Build JIT-compiled JAX sampler for one reverse insert/delete step."""
        max_rejects = self.max_rejects

        def _single_reverse_sample(
            was_missing: jax.Array,
            was_insert_log_prob: jax.Array,
            previous_token_log_probs: jax.Array,
            delete_log_probs: jax.Array,
            length: jax.Array,
            rng: jax.Array,
        ) -> gfp.DynamicLengthSentinelSequence:
            marginals = gfp.ReverseProcessMarginalDistribution(
                was_missing=was_missing,
                was_insert_log_prob=was_insert_log_prob,
                previous_token_log_probs=previous_token_log_probs,
                log_prob_number_of_preceding_deletes=delete_log_probs,
                length=length,
            )

            def _generate_sample(key):
                seq = marginals.sample(key)
                return seq.is_valid(), seq

            return gutil.rejection_sample(
                _generate_sample, rng, max_rejects=max_rejects
            )

        self._jax_sample_reverse_batch = jax.jit(
            jax.vmap(_single_reverse_sample, in_axes=(0, 0, 0, 0, 0, 0))
        )

    def _maybe_apply_model_parameterization(
        self,
        *,
        pred_insert_logprob: torch.Tensor,
        pred_prev_token_logprob: torch.Tensor,
        pred_delete_logprob: torch.Tensor,
        xtplus1_tokens: torch.Tensor,
        xtplus1_length: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert model outputs to reverse-step parameterization when needed.

        Args:
            pred_insert_logprob: Approximate insert log-probs, shape [B, L].
            pred_prev_token_logprob: Approximate previous-token log-probs, shape [B, L, K].
            pred_delete_logprob: Approximate delete-count log-probs, shape [B, L+1, D].
            xtplus1_tokens: Current sequence tokens in Google sentinel space, shape [B, L].
            xtplus1_length: Current sequence lengths, shape [B].
            t: Per-sample diffusion step indices, shape [B].

        Returns:
            Tuple of tensors `(insert_lp, prev_token_lp, delete_lp)` in the reverse
            process parameterization expected by loss/sampling.
        """
        if self.model_prediction != "x0":
            return pred_insert_logprob, pred_prev_token_logprob, pred_delete_logprob
        if self.x0_parameterizer is None:
            raise RuntimeError("x0 parameterizer is not initialized.")
        return self.x0_parameterizer.apply_torch(
            approx_was_insert_log_prob=pred_insert_logprob,
            approx_previous_token_log_probs=pred_prev_token_logprob,
            approx_delete_log_probs=pred_delete_logprob,
            xtplus1_tokens=xtplus1_tokens.to(torch.int32),
            xtplus1_length=xtplus1_length.to(torch.int32),
            t=t.to(torch.int32),
        )

    def _prepare_tokens_for_x0_parameterization(
        self, xtplus1_tokens: np.ndarray, xtplus1_length: np.ndarray
    ) -> np.ndarray:
        """Bridge wrapper: replace padded ERROR slots with DEL sentinels."""
        return self.bridge.prepare_tokens_for_x0_parameterization(
            xtplus1_tokens, xtplus1_length
        )

    def _jax_tokens_lengths_to_torch_batch(
        self, tokens: np.ndarray, lengths: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Bridge wrapper: convert Google sentinel arrays to Torch token ids."""
        return self.bridge.jax_tokens_lengths_to_torch_batch(
            tokens, lengths, device=device
        )

    def _attach_virtual_eos_boundary_for_runner(
        self, tokens: torch.Tensor, lengths: np.ndarray
    ) -> torch.Tensor:
        """Insert EOS marker for runner compatibility after reverse sampling.

        Insert/delete diffusion itself does not model EOS as a stochastic token.
        The base evaluation pipeline in this repo expects one EOS marker per
        sequence, so we stamp EOS at the predicted boundary index post-hoc.
        """
        out = tokens.clone()
        bsz, seq_len = out.shape
        eos_idx = torch.tensor(lengths, dtype=torch.long, device=out.device).clamp(
            min=0, max=max(seq_len - 1, 0)
        )
        out.scatter_(1, eos_idx[:, None], self.eos_token_id)
        pos = torch.arange(seq_len, device=out.device)[None, :]
        out = torch.where(
            pos > eos_idx[:, None], torch.full_like(out, self.pad_token_id), out
        )
        return out

    def _build_decoder_input_with_eos_boundary(
        self, xt: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Append an explicit EOS-boundary position for decoder conditioning.

        Args:
            xt: Current noised tokens `[B, L]` in Torch token space.
            lengths: True sequence lengths `[B]` for `xt`.

        Returns:
            Tensor `[B, L+1]` where position `length` is set to EOS token id.
            This mirrors the paper's EOS-boundary delete-count prediction.
        """
        bsz, seq_len = xt.shape
        xt_model = torch.full(
            (bsz, seq_len + 1),
            self.pad_token_id,
            dtype=xt.dtype,
            device=xt.device,
        )
        xt_model[:, :seq_len] = xt
        eos_idx = lengths.to(torch.long).clamp(min=0, max=seq_len)
        xt_model.scatter_(1, eos_idx[:, None], self.eos_token_id)
        return xt_model

    def _build_batch_targets(
        self, x0: torch.Tensor, token_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Sample `(x_t, alignment targets)` used by insert/delete ELBO training.

        Args:
            x0: Clean target tokens in Torch vocabulary, shape [B, L].
            token_mask: True-token mask (PAD excluded), shape [B, L].

        Returns:
            Dictionary containing noised sequence, timestep info, validity mask and
            per-component target log-prob tensors from Google forward-process math.
        """
        bsz = x0.shape[0]
        x0_batch = self.bridge.torch_x0_to_jax_dynamic_sequence(x0, token_mask)
        keys = self._split_key(num=bsz)
        jax_targets = self._jax_batched_targets(x0_batch, keys)
        batch_targets = jax.device_get(jax_targets)
        return self.bridge.jax_targets_to_torch(batch_targets, device=x0.device)

    def _denoise_prediction_fn(
        self,
        *,
        backbone,
        t: torch.Tensor,
        xtplus1: torch.Tensor,
        xtplus1_tokens_jax: torch.Tensor,
        xtplus1_length: torch.Tensor,
        model_kwargs: Dict,
        return_token_logits_full: bool = False,
    ) -> (
        tuple[ReverseProcessMarginalDistributionTorch, torch.Tensor]
        | tuple[ReverseProcessMarginalDistributionTorch, torch.Tensor, torch.Tensor]
    ):
        """Torch translation of `training_setup.py: denoise_prediction_fn(...)`.

        This produces the two model outputs used in the ELBO term:
        1) reverse-process marginals over `(insert, previous-token, delete-count)`
        2) `p_theta(|x_T|)` length log-probabilities.
        """
        xt_lengths = xtplus1_length.to(torch.long)
        xt_model = self._build_decoder_input_with_eos_boundary(xtplus1, xt_lengths)
        eos_idx = xt_lengths.clamp(min=0, max=xtplus1.shape[1])

        local_kwargs = dict(model_kwargs)
        t_norm = t.float() / max(self.num_steps - 1, 1)
        local_kwargs["timesteps"] = (
            t_norm if self.time_conditioning else torch.zeros_like(t_norm)
        )
        if self.self_condition and "self_conditions" not in local_kwargs:
            local_kwargs["self_conditions"] = torch.zeros(
                xt_model.shape[0],
                xt_model.shape[1],
                backbone.predcats,
                device=xtplus1.device,
            )

        model_out = backbone(xt_model, **local_kwargs, return_hidden=True)
        token_logits_full = model_out["out"][:, : xtplus1.shape[1]]
        hidden_all = model_out["hidden"]
        hidden_tokens = hidden_all[:, : xtplus1.shape[1]]
        hidden_eos = hidden_all.gather(
            1, eos_idx[:, None, None].expand(-1, 1, hidden_all.shape[-1])
        ).squeeze(1)

        real_ids = self.real_torch_ids_tensor.to(xtplus1.device)
        token_logits = token_logits_full.index_select(dim=-1, index=real_ids)
        insert_logits = backbone.insertdelete_insert_head(hidden_tokens).squeeze(-1)
        delete_logits = backbone.insertdelete_delete_head(hidden_tokens)
        eos_delete_logits = backbone.insertdelete_delete_head(hidden_eos)
        pooled_hidden = hidden_tokens.mean(dim=1)
        length_prediction = F.log_softmax(
            backbone.insertdelete_length_head(pooled_hidden), dim=-1
        )

        all_logits = [
            token_logits,
            insert_logits,
            delete_logits,
            eos_delete_logits,
        ]

        # check for nans in logits
        for logit in all_logits:
            if torch.isnan(logit).any():
                raise ValueError("NaN detected in model logits.")

        pred_insert_logprob = F.logsigmoid(insert_logits)
        pred_not_insert_logprob = F.logsigmoid(-insert_logits)
        pred_token_cond_logprob = F.log_softmax(token_logits, dim=-1)
        pred_prev_token_logprob = (
            pred_not_insert_logprob[..., None] + pred_token_cond_logprob
        )
        pred_delete_logprob = F.log_softmax(
            torch.cat([delete_logits, eos_delete_logits[:, None, :]], dim=1),
            dim=-1,
        )

        xt_tokens_for_param = xtplus1_tokens_jax.to(torch.int32)
        xt_lengths_for_param = xtplus1_length.to(torch.int32)
        if self.model_prediction == "x0":
            xt_tokens_np = self._prepare_tokens_for_x0_parameterization(
                xtplus1_tokens_jax.detach().cpu().numpy().astype(np.int32),
                xtplus1_length.detach().cpu().numpy().astype(np.int32),
            )
            xt_tokens_for_param = torch.tensor(
                xt_tokens_np, dtype=torch.int32, device=xtplus1.device
            )
            xt_lengths_for_param = xtplus1_length.to(torch.int32)

        pred_insert_logprob, pred_prev_token_logprob, pred_delete_logprob = (
            self._maybe_apply_model_parameterization(
                pred_insert_logprob=pred_insert_logprob,
                pred_prev_token_logprob=pred_prev_token_logprob,
                pred_delete_logprob=pred_delete_logprob,
                xtplus1_tokens=xt_tokens_for_param,
                xtplus1_length=xt_lengths_for_param,
                t=t.to(torch.int32),
            )
        )

        was_missing = xtplus1_tokens_jax.eq(
            gfp.DynamicLengthSentinelSequence.INSERT_SENTINEL
        )
        denoise_prediction = ReverseProcessMarginalDistributionTorch(
            was_missing=was_missing,
            was_insert_log_prob=pred_insert_logprob,
            previous_token_log_probs=pred_prev_token_logprob,
            log_prob_number_of_preceding_deletes=pred_delete_logprob,
            length=xtplus1_length.to(torch.long),
        )
        if return_token_logits_full:
            return denoise_prediction, length_prediction, token_logits_full
        return denoise_prediction, length_prediction

    def _sample_based_elbo_term_loss(
        self,
        *,
        backbone,
        x0: torch.Tensor,
        model_kwargs: Dict,
        token_mask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Torch translation of Google `sample_based_elbo_term_loss`.

        We keep Google's variable names/order where possible:
        - sample `targets` from forward process (`q`)
        - run model (`denoise_prediction_fn`)
        - compute reverse-step cross-entropy term
        - apply timestep importance correction
        - add terminal-length prior term.
        """
        targets = self._build_batch_targets(x0, token_mask)
        denoise_targets = ReverseProcessMarginalDistributionTorch(
            was_missing=targets["target_was_missing"],
            was_insert_log_prob=targets["target_was_insert_logprob"],
            previous_token_log_probs=targets["target_prev_token_logprob"],
            log_prob_number_of_preceding_deletes=targets["target_delete_logprob"],
            length=targets["target_length"],
        )
        denoise_prediction, length_prediction = self._denoise_prediction_fn(
            backbone=backbone,
            t=targets["t"],
            xtplus1=targets["xt"],
            xtplus1_tokens_jax=targets["xt_tokens_jax"],
            xtplus1_length=targets["xt_length"],
            model_kwargs=model_kwargs,
        )

        _, extra_cross_entropy_info = (
            ReverseProcessMarginalDistributionTorch.cross_entropy(
                samples_from=denoise_targets,
                model_output=denoise_prediction,
                return_extra=True,
            )
        )
        expected_p_xt_given_xtplus1 = (
            self.loss_weight_delete * extra_cross_entropy_info["delete_term"]
            + self.loss_weight_insert * extra_cross_entropy_info["insert_term"]
            + self.loss_weight_token * extra_cross_entropy_info["token_term"]
        )
        expected_q_xtplus1_given_xt = targets["expected_q"]
        elbo_term_at_t = expected_p_xt_given_xtplus1 - expected_q_xtplus1_given_xt
        elbo_across_terms_estimate = elbo_term_at_t / targets["t_weight"].clamp_min(
            1e-8
        )

        lp_of_xT_length = torch.sum(
            safe_exp_weighted_torch(
                targets["target_length_logprob"], length_prediction
            ),
            dim=1,
        )
        lp_of_xT_length = self.loss_weight_length * lp_of_xT_length
        elbo_estimate = elbo_across_terms_estimate + lp_of_xT_length

        results = {
            "neg_elbo_timestep_term": -elbo_term_at_t,
            "nll_length": -lp_of_xT_length,
            "neg_elbo_total_importance_sample": -elbo_estimate,
            "expected_p_xt_given_xtplus1": expected_p_xt_given_xtplus1,
            "expected_q_xtplus1_given_xt": expected_q_xtplus1_given_xt,
            **extra_cross_entropy_info,
        }
        valid = (targets["valid"] > 0.5).to(elbo_estimate.dtype)
        results = {
            key: torch.where(valid > 0, value, torch.zeros_like(value))
            for key, value in results.items()
        }
        return results, valid, {"t": targets["t"], "xt": targets["xt"]}

    def insertdelete_training_loss(
        self,
        backbone,
        x0: torch.Tensor,
        model_kwargs: Dict,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute insert/delete ELBO loss (Torch translation of Google training path)."""
        if token_mask is None:
            token_mask = x0 != self.pad_token_id

        results, denominator, extras = self._sample_based_elbo_term_loss(
            backbone=backbone,
            x0=x0,
            model_kwargs=model_kwargs,
            token_mask=token_mask,
        )
        valid_count = denominator.sum().clamp_min(1.0)
        loss_per_sample = results["neg_elbo_total_importance_sample"]
        loss = loss_per_sample.sum() / valid_count

        return {
            "loss_per_sample": loss_per_sample,
            "loss": loss,
            "valid_count": valid_count,
            "elbo_timestep_term": -results["neg_elbo_timestep_term"],
            "length_term": -results["nll_length"],
            "expected_p": results["expected_p_xt_given_xtplus1"],
            "expected_q": results["expected_q_xtplus1_given_xt"],
            "delete_term": results["delete_term"],
            "insert_term": results["insert_term"],
            "token_term": results["token_term"],
            **extras,
        }

    @torch.no_grad()
    def _sample(
        self,
        x=None,
        num_steps=None,
        eps=1e-5,
        top=None,
        model_kwargs={},
        save_x=False,
        save_p=False,
        progress=False,
    ):
        """Sample sequences from the insert/delete reverse process.

        Args:
            x: Optional initial sequence batch (unused for standard ancestral sampling).
            num_steps: Reverse steps to run; must equal configured `T`.
            eps: Unused placeholder for API compatibility.
            top: Unused placeholder for API compatibility.
            model_kwargs: Decoder conditioning inputs.
            save_x: If True, return intermediate x states.
            save_p: If True, return intermediate token probabilities.
            progress: Unused placeholder for API compatibility.

        Returns:
            Dict with final `prediction`, final `logits`, and optional trajectories.
        """
        del eps, progress
        steps = self.num_steps if num_steps is None else int(num_steps)
        if steps != self.num_steps:
            raise ValueError(
                "Insert/delete sampler currently requires sampling.steps == T for "
                "a mathematically faithful reverse chain."
            )

        if x is None:
            batch_size = (
                len(model_kwargs["charge"])
                if model_kwargs.get("charge") is not None
                else model_kwargs["kv_features"].shape[0]
            )
            run_device = model_kwargs["kv_features"].device
        else:
            batch_size = x.shape[0]
            run_device = x.device

        # Estimate p_theta(|x_T|) from an all-DEL probe state.
        probe_x = torch.full(
            (batch_size, self.max_len),
            self.delete_token_id,
            dtype=torch.long,
            device=run_device,
        )
        probe_kwargs = dict(model_kwargs)
        probe_kwargs["timesteps"] = torch.ones(
            (batch_size,), dtype=torch.float32, device=run_device
        )
        if self.self_condition and "self_conditions" not in probe_kwargs:
            probe_kwargs["self_conditions"] = torch.zeros(
                batch_size, self.max_len, self.backbone.predcats, device=run_device
            )
        probe_out = self.backbone(probe_x, **probe_kwargs, return_hidden=True)
        probe_hidden = probe_out["hidden"]
        length_logits = self.backbone.insertdelete_length_head(probe_hidden.mean(dim=1))
        length_dist = torch.distributions.Categorical(logits=length_logits)
        xt_lengths_torch = length_dist.sample()
        xt_lengths = (
            xt_lengths_torch.detach().cpu().numpy().astype(np.int32, copy=False)
        )

        xt_tokens = np.full(
            (batch_size, self.max_len),
            gfp.DynamicLengthSentinelSequence.ERROR,
            dtype=np.int32,
        )
        active = np.arange(self.max_len)[None, :] < xt_lengths[:, None]
        xt_tokens[active] = gfp.DynamicLengthSentinelSequence.DELETE_SENTINEL

        x_save_list = []
        p_save_list = []
        final_logits = None
        keys_per_step = self._split_key(num=batch_size * self.num_steps).reshape(
            self.num_steps, batch_size, 2
        )

        for t in range(self.num_steps - 1, -1, -1):
            xt_torch = self._jax_tokens_lengths_to_torch_batch(
                xt_tokens, xt_lengths, run_device
            )
            xt_lengths_torch = torch.tensor(
                xt_lengths, dtype=torch.long, device=run_device
            )
            t_batch = torch.full(
                (batch_size,),
                int(t),
                dtype=torch.long,
                device=run_device,
            )
            denoise_prediction, _, token_logits_full = self._denoise_prediction_fn(
                backbone=self.backbone,
                t=t_batch,
                xtplus1=xt_torch,
                xtplus1_tokens_jax=torch.tensor(
                    xt_tokens, dtype=torch.int32, device=run_device
                ),
                xtplus1_length=xt_lengths_torch.to(torch.int32),
                model_kwargs=model_kwargs,
                return_token_logits_full=True,
            )
            final_logits = token_logits_full

            was_missing = denoise_prediction.was_missing.detach().cpu().numpy()
            sampled_xt = self._jax_sample_reverse_batch(
                jnp.asarray(was_missing, dtype=jnp.bool_),
                jnp.asarray(
                    denoise_prediction.was_insert_log_prob.detach().cpu().numpy(),
                    dtype=jnp.float32,
                ),
                jnp.asarray(
                    denoise_prediction.previous_token_log_probs.detach().cpu().numpy(),
                    dtype=jnp.float32,
                ),
                jnp.asarray(
                    denoise_prediction.log_prob_number_of_preceding_deletes.detach()
                    .cpu()
                    .numpy(),
                    dtype=jnp.float32,
                ),
                jnp.asarray(xt_lengths, dtype=jnp.int32),
                keys_per_step[t],
            )
            xt_tokens = np.asarray(jax.device_get(sampled_xt.tokens), dtype=np.int32)
            xt_lengths = np.asarray(jax.device_get(sampled_xt.length), dtype=np.int32)

            if save_x:
                x_save_list.append(
                    self._jax_tokens_lengths_to_torch_batch(
                        xt_tokens, xt_lengths, run_device
                    )
                )
            if save_p:
                p_save_list.append(token_logits_full.softmax(dim=-1))

        prediction = self._jax_tokens_lengths_to_torch_batch(
            xt_tokens, xt_lengths, run_device
        )
        prediction = self._attach_virtual_eos_boundary_for_runner(
            prediction, xt_lengths
        )
        result = {"prediction": prediction, "logits": final_logits}
        if save_x:
            result["x_save"] = torch.stack(x_save_list, dim=1)
        if save_p:
            result["p_save"] = torch.stack(p_save_list, dim=1)
        return result
