from __future__ import annotations

from typing import Dict, Tuple

import jax
import numpy as np
import torch

from google_d3pm_ins_del import forward_process as gfp


def _safe_logaddexp_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """`logaddexp` with zero-gradient behavior when both branches are impossible."""
    impossible = torch.isneginf(torch.maximum(x1, x2))
    x1_safe = torch.where(impossible, x1.detach(), x1)
    x2_safe = torch.where(impossible, x2.detach(), x2)
    return torch.logaddexp(x1_safe, x2_safe)


def _safe_logsumexp_torch(a: torch.Tensor, dim: int) -> torch.Tensor:
    """`logsumexp` with zero-gradient behavior when all values are impossible."""
    max_along_dim = torch.amax(a, dim=dim, keepdim=True)
    impossible = torch.isneginf(max_along_dim)
    a_safe = torch.where(impossible, a.detach(), a)
    return torch.logsumexp(a_safe, dim=dim)


def _safe_sub_or_ninf_torch(
    case_log_prob: torch.Tensor, total_log_prob: torch.Tensor
) -> torch.Tensor:
    """Return `case-total`, or `-inf` where `total` is impossible."""
    neg_inf = torch.full_like(total_log_prob, float("-inf"))
    return torch.where(
        torch.isneginf(total_log_prob),
        neg_inf,
        case_log_prob - total_log_prob,
    )


def _apply_log_matrix_to_log_distn(
    before_log_probs: torch.Tensor, log_matrix: torch.Tensor
) -> torch.Tensor:
    """Torch analogue of `A.apply(before, is_distn=True, log=True)`."""
    terms = before_log_probs[..., :, None] + log_matrix[:, None, :, :]
    return _safe_logsumexp_torch(terms, dim=-2)


def _observe_log_matrix_on_tokens(
    log_matrix: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Torch analogue of `vmap(A.observe(token, is_distn=False, log=True))`.

    Args:
        log_matrix: `[B, K, K]`, where rows are previous-token ids and columns
            are observed-token ids.
        token_ids: `[B, L]` observed token ids in `[0, K-1]`.

    Returns:
        `[B, L, K]` log-likelihoods of each previous token for each observation.
    """
    k = log_matrix.shape[-1]
    gathered = torch.gather(
        log_matrix,
        dim=2,
        index=token_ids[:, None, :].expand(-1, k, -1),
    )
    return gathered.transpose(1, 2)


def _apply_x0_parameterization_single_torch(
    approximate_x0_guess: Dict[str, torch.Tensor],
    xtplus1: Dict[str, torch.Tensor],
    d_0_to_t: Dict[str, torch.Tensor],
    d_t_to_tplus1: Dict[str, torch.Tensor],
    precomputed_Attplus1_xtplus1: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Torch translation of Google `forward_process.apply_x0_parameterization`.

    This mirrors Google variable names and math flow, but in Torch so gradients
    flow naturally from model outputs.
    """
    max_len = xtplus1["tokens"].shape[1]
    vocab_size = approximate_x0_guess["previous_token_log_probs"].shape[-1]

    if precomputed_Attplus1_xtplus1 is None:
        xt_mod = torch.remainder(xtplus1["tokens"], vocab_size)
        precomputed_Attplus1_xtplus1 = _observe_log_matrix_on_tokens(
            d_t_to_tplus1["A_log_matrix"], xt_mod
        )

    # Sketch:
    # If model predicts something is part of x0:
    # - Token comes from the standard A1.apply + A2.observe
    # - Number of inserts is based on taking its number of inserts prediction
    #   and doing a non-rerolling binomial
    # If model predicts something is NOT part of x0:
    # - Token comes from the silent insert distribution
    # - Number of inserts is based on taking its number of inserts prediction
    #   and doing a non-rerolling binomial, PLUS shifting it by one geometric
    #   to account for the things inserted before this token.
    # Why does this make sense? Well, if model says not part of x0, then it's
    # either a reroll or a non-reroll insert. Non-reroll inserts are just
    # inserts that don't have a deletion before them, and the difference in the
    # number of DEL sentinels that get added in the two cases is just how many
    # things in x0 were deleted before we got there.
    distn_deletes_at_t_from_deletes_at_x0 = d_0_to_t[
        "delete_count_marginal_log_probs"
    ]

    def process_token(
        insert_logprob: torch.Tensor,
        x0_token_distn: torch.Tensor,
        xtplus1_token: torch.Tensor,
        precomputed_Attplus1_xtplus1_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # How likely to see xtplus1_token from each other token?
        observe_from_token_logits = torch.where(
            xtplus1_token[..., None].eq(gfp.DynamicLengthSentinelSequence.DELETE_SENTINEL),
            d_t_to_tplus1["lp_delete"][:, None, None],
            precomputed_Attplus1_xtplus1_logits,
        )
        # How likely to see xtplus1_token from an insert at time t?
        xt_mod = torch.remainder(xtplus1_token, vocab_size)
        observe_from_insert_logit = torch.where(
            xtplus1_token.eq(gfp.DynamicLengthSentinelSequence.DELETE_SENTINEL),
            d_t_to_tplus1["lp_delete"][:, None],
            torch.gather(d_t_to_tplus1["D_insert_logits"], 1, xt_mod),
        )
        # Given that there was an insert, when did it happen?
        total_insert_lp = _safe_logaddexp_torch(
            d_0_to_t["lp_sentinel_insert"],
            d_0_to_t["lp_silent_insert"],
        )
        lp_insert_is_sentinel = _safe_sub_or_ninf_torch(
            d_0_to_t["lp_sentinel_insert"], total_insert_lp
        )
        lp_insert_is_silent = _safe_sub_or_ninf_torch(
            d_0_to_t["lp_silent_insert"], total_insert_lp
        )

        # Suppose this was a token from x0, according to our guess.
        xt_token_logits_from_keep = (
            _apply_log_matrix_to_log_distn(x0_token_distn, d_0_to_t["A_log_matrix"])
            + observe_from_token_logits
        )

        # Suppose this was an insert, and suppose it happened at time t.
        xt_insert_logit_from_insert = (
            insert_logprob + lp_insert_is_sentinel[:, None] + observe_from_insert_logit
        )

        # Suppose this was an insert, but it happened before time t.
        xt_token_logits_from_insert = (
            insert_logprob[..., None]
            + lp_insert_is_silent[:, None, None]
            + d_0_to_t["D_silent_insert_logits"][:, None, :]
            + observe_from_token_logits
        )

        # Combine our cases.
        xt_token_logits = _safe_logaddexp_torch(
            xt_token_logits_from_keep, xt_token_logits_from_insert
        )
        xt_insert_logit = xt_insert_logit_from_insert
        denominator = _safe_logaddexp_torch(
            xt_insert_logit,
            _safe_logsumexp_torch(xt_token_logits, dim=-1),
        )
        xt_token_logits_normalized = xt_token_logits - denominator[..., None]
        xt_insert_logit_normalized = xt_insert_logit - denominator

        return xt_token_logits_normalized, xt_insert_logit_normalized

    token_logits, insert_logit = process_token(
        approximate_x0_guess["was_insert_log_prob"],
        approximate_x0_guess["previous_token_log_probs"],
        xtplus1["tokens"],
        precomputed_Attplus1_xtplus1,
    )

    # In any of these situations, there's always at least one chance to add
    # geometric-r.v.s of ephemeral insert-to-delete sentinels. If we think there
    # were more tokens between this and the previous in x0, we may have more.
    # We let the model guess how many were in x0, and compute how many we'd
    # see here as a function of that. It's likely correlated with whether or
    # not this was an insert, and where it was, but due to conditional
    # independence assumptions we ignore those interactions.
    #
    # Note: We're folding rerolling into a special case of "having deletions
    # before an insert".
    def adjust_deletes(num_deletes_before: torch.Tensor) -> torch.Tensor:
        terms = (
            num_deletes_before[:, :, :, None]
            + distn_deletes_at_t_from_deletes_at_x0[:, None, :, :]
        )
        return _safe_logsumexp_torch(terms, dim=-2)

    deletes_adjusted = adjust_deletes(
        approximate_x0_guess["log_prob_number_of_preceding_deletes"]
    )

    return {
        "was_missing": approximate_x0_guess["was_missing"],
        "was_insert_log_prob": insert_logit,
        "previous_token_log_probs": token_logits,
        "log_prob_number_of_preceding_deletes": deletes_adjusted,
        "length": approximate_x0_guess["length"],
    }


class InsertDeleteX0Parameterizer:
    """x0-parameterization transform using Torch autodiff-friendly math."""

    def __init__(self, schedule) -> None:
        self.schedule = schedule
        self._stacked_np_const_cache: Dict[int, Dict[str, np.ndarray]] = {}

    @staticmethod
    def _to_torch_const(
        value,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(np.array(value, copy=True), device=device, dtype=dtype)

    def _build_stacked_np_constants(self, max_len: int) -> Dict[str, np.ndarray]:
        """Prestack per-step constants so runtime can gather by batch timestep."""
        if max_len in self._stacked_np_const_cache:
            return self._stacked_np_const_cache[max_len]

        t_steps = int(self.schedule.num_steps)
        a0t = []
        attp1 = []
        lp_del = []
        d_ins = []
        lp_sent_ins = []
        lp_silent_ins = []
        d_silent_ins = []
        del_marg = []

        for t in range(t_steps):
            d_0_to_t, d_t_to_tplus1 = self.schedule.distns_at_step(t)
            delete_count_marginals, _ = d_0_to_t.delete_count_marginals(max_len)
            a0t.append(
                np.asarray(jax.device_get(d_0_to_t.A.prob_matrix(log=True)), dtype=np.float32)
            )
            attp1.append(
                np.asarray(
                    jax.device_get(d_t_to_tplus1.A.prob_matrix(log=True)),
                    dtype=np.float32,
                )
            )
            lp_del.append(float(jax.device_get(d_t_to_tplus1.lp_delete)))
            d_ins.append(
                np.asarray(jax.device_get(d_t_to_tplus1.D_insert_logits), dtype=np.float32)
            )
            lp_sent_ins.append(float(jax.device_get(d_0_to_t.lp_sentinel_insert)))
            lp_silent_ins.append(float(jax.device_get(d_0_to_t.lp_silent_insert)))
            d_silent_ins.append(
                np.asarray(jax.device_get(d_0_to_t.D_silent_insert_logits), dtype=np.float32)
            )
            del_marg.append(
                np.asarray(jax.device_get(delete_count_marginals.log_probs), dtype=np.float32)
            )

        stacked = {
            "a0t_log_matrix": np.stack(a0t, axis=0),
            "attplus1_log_matrix": np.stack(attp1, axis=0),
            "lp_delete": np.asarray(lp_del, dtype=np.float32),
            "d_insert_logits": np.stack(d_ins, axis=0),
            "lp_sentinel_insert": np.asarray(lp_sent_ins, dtype=np.float32),
            "lp_silent_insert": np.asarray(lp_silent_ins, dtype=np.float32),
            "d_silent_insert_logits": np.stack(d_silent_ins, axis=0),
            "delete_count_marginal_log_probs": np.stack(del_marg, axis=0),
        }
        self._stacked_np_const_cache[max_len] = stacked
        return stacked

    def _get_stacked_torch_constants(
        self,
        *,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        np_stacked = self._build_stacked_np_constants(max_len=max_len)
        return {
            key: self._to_torch_const(value, device=device, dtype=dtype)
            for key, value in np_stacked.items()
        }

    def apply_torch(
        self,
        *,
        approx_was_insert_log_prob: torch.Tensor,
        approx_previous_token_log_probs: torch.Tensor,
        approx_delete_log_probs: torch.Tensor,
        xtplus1_tokens: torch.Tensor,
        xtplus1_length: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized batch wrapper around single-step x0 parameterization."""
        if approx_was_insert_log_prob.ndim != 2:
            raise ValueError("Expected approx_was_insert_log_prob with shape [B, L].")
        if approx_previous_token_log_probs.ndim != 3:
            raise ValueError("Expected approx_previous_token_log_probs with shape [B, L, K].")
        if approx_delete_log_probs.ndim != 3:
            raise ValueError("Expected approx_delete_log_probs with shape [B, L+1, D].")

        bsz, max_len = approx_was_insert_log_prob.shape
        if xtplus1_tokens.shape != (bsz, max_len):
            raise ValueError(
                f"xtplus1_tokens must have shape {(bsz, max_len)}, got {tuple(xtplus1_tokens.shape)}."
            )

        consts = self._get_stacked_torch_constants(
            max_len=max_len,
            device=approx_was_insert_log_prob.device,
            dtype=approx_was_insert_log_prob.dtype,
        )
        t_idx = t.to(torch.long)

        d_0_to_t = {
            "A_log_matrix": consts["a0t_log_matrix"].index_select(0, t_idx),
            "lp_sentinel_insert": consts["lp_sentinel_insert"].index_select(0, t_idx),
            "lp_silent_insert": consts["lp_silent_insert"].index_select(0, t_idx),
            "D_silent_insert_logits": consts["d_silent_insert_logits"].index_select(
                0, t_idx
            ),
            "delete_count_marginal_log_probs": consts[
                "delete_count_marginal_log_probs"
            ].index_select(0, t_idx),
        }
        d_t_to_tplus1 = {
            "A_log_matrix": consts["attplus1_log_matrix"].index_select(0, t_idx),
            "lp_delete": consts["lp_delete"].index_select(0, t_idx),
            "D_insert_logits": consts["d_insert_logits"].index_select(0, t_idx),
        }

        approximate_x0_guess = {
            "was_missing": xtplus1_tokens.eq(gfp.DynamicLengthSentinelSequence.INSERT_SENTINEL),
            "was_insert_log_prob": approx_was_insert_log_prob,
            "previous_token_log_probs": approx_previous_token_log_probs,
            "log_prob_number_of_preceding_deletes": approx_delete_log_probs,
            "length": xtplus1_length,
        }
        xtplus1 = {
            "tokens": xtplus1_tokens.to(torch.long),
            "length": xtplus1_length.to(torch.long),
        }
        adjusted = _apply_x0_parameterization_single_torch(
            approximate_x0_guess=approximate_x0_guess,
            xtplus1=xtplus1,
            d_0_to_t=d_0_to_t,
            d_t_to_tplus1=d_t_to_tplus1,
            precomputed_Attplus1_xtplus1=None,
        )
        return (
            adjusted["was_insert_log_prob"],
            adjusted["previous_token_log_probs"],
            adjusted["log_prob_number_of_preceding_deletes"],
        )
