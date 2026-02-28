import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _build_beta_schedule(num_steps: int, schedule_cfg: Dict) -> torch.Tensor:
    schedule_type = schedule_cfg.get("beta_schedule", "cosine")
    dtype = torch.float64

    if "betas" in schedule_cfg and schedule_cfg["betas"] is not None:
        betas = torch.as_tensor(schedule_cfg["betas"], dtype=dtype)
        if betas.numel() != num_steps:
            raise ValueError(f"Expected {num_steps} betas, got {betas.numel()}.")
        return betas

    if schedule_type == "linear":
        beta_start = float(schedule_cfg.get("beta_start", 1e-4))
        beta_end = float(schedule_cfg.get("beta_end", 2e-2))
        return torch.linspace(beta_start, beta_end, num_steps, dtype=dtype)

    if schedule_type == "cosine":
        # ref repo cloneofsimo/d3pm_runner.py uses this cosine-derived schedule.
        # ref repo google-images accepts precomputed betas and is agnostic here.
        steps = torch.arange(num_steps + 1, dtype=dtype) / num_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.minimum(betas, torch.full_like(betas, 0.999))

    raise NotImplementedError(f"Unknown beta schedule '{schedule_type}'.")


def build_q_matrices(
    num_classes: int,
    num_steps: int,
    transition_cfg: Dict,
    *,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """Build Q_t and cumulative Qbar_t.

    Returns:
      q_one_step: [T, K, K]
      q_cumulative: [T, K, K]
      q_one_step_transposed: [T, K, K]
      betas: [T]
    """
    betas = _build_beta_schedule(num_steps, transition_cfg)
    transition_type = transition_cfg.get("type", "uniform")

    q_one_step_mats = []
    for beta in betas:
        if transition_type == "uniform":
            # ref repo google-images _get_full_transition_mat uses this row-stochastic form.
            # ref repo insertdelete builds operators (then/apply/observe) instead of raw [K,K] tensors.
            mat = torch.full(
                (num_classes, num_classes), beta / float(num_classes), dtype=dtype
            )
            diag_val = 1.0 - beta * (num_classes - 1.0) / num_classes
            mat.diagonal().fill_(diag_val)
        elif transition_type == "absorbing":
            # ref repo google-images _get_absorbing_transition_mat adds beta mass to absorbing column.
            # ref repo cloneofsimo default runner does not implement this transition type.
            absorbing_idx = int(transition_cfg.get("absorbing_idx", num_classes // 2))
            mat = torch.diag(torch.full((num_classes,), 1.0 - beta, dtype=dtype))
            mat[:, absorbing_idx] = mat[:, absorbing_idx] + beta
        elif transition_type == "explicit":
            # ref repo insertdelete transition_operator.MatrixOperator accepts explicit matrices.
            # Divergence: here we support either [K,K] (shared) or [T,K,K] tensors directly.
            explicit = torch.as_tensor(transition_cfg["matrix"], dtype=dtype)
            if explicit.ndim == 2:
                mat = explicit.clone()
            elif explicit.ndim == 3:
                step_index = len(q_one_step_mats)
                mat = explicit[step_index].clone()
            else:
                raise ValueError("explicit transition matrix must have ndim 2 or 3")
        else:
            raise NotImplementedError(f"Unknown transition type '{transition_type}'.")

        row_sums = mat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mat = mat / row_sums
        q_one_step_mats.append(mat)

    q_one_step = torch.stack(q_one_step_mats, dim=0)

    # ref repo google-images builds cumulative Qbar_t via chained matrix products.
    # ref repo cloneofsimo does the same (q_mat_t = q_mat_t @ q_onestep[idx]).
    q_cumulative = []
    q_running = q_one_step[0]
    q_cumulative.append(q_running)
    for t in range(1, num_steps):
        q_running = q_running @ q_one_step[t]
        q_cumulative.append(q_running)
    q_cumulative = torch.stack(q_cumulative, dim=0)

    q_one_step_transposed = q_one_step.transpose(1, 2)

    return {
        "q_one_step": q_one_step,
        "q_cumulative": q_cumulative,
        "q_one_step_transposed": q_one_step_transposed,
        "betas": betas.to(dtype),
    }


def _at(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # ref repo google-images _at uses a[t_broadcast, x].
    # ref repo cloneofsimo _at is equivalent but uses 1-based t indexing.
    bsz = x.shape[0]
    t_view = t.view(bsz, *([1] * (x.ndim - 1)))
    return a[t_view, x, :]


def _at_onehot(a: torch.Tensor, t: torch.Tensor, x_probs: torch.Tensor) -> torch.Tensor:
    # ref repo google-images _at_onehot does a batched matmul with a[t].
    # Divergence: sequence case is fixed to [B,L,K] and uses einsum for clarity.
    a_t = a[t]
    return torch.einsum("blk,bkd->bld", x_probs, a_t)


def _q_transition_s_to_t(
    q_one_step: torch.Tensor, s: int, t: int
) -> torch.Tensor:
    """Compute q(x_t | x_s) transition matrix for any s <= t.

    This is used for jump-aware sampling on sparse timestep grids. For adjacent
    steps (s=t-1), this reduces to q_one_step[t].
    """
    if s > t:
        raise ValueError(f"Expected s <= t, got s={s}, t={t}.")

    num_classes = q_one_step.shape[-1]
    if s == t:
        return torch.eye(num_classes, dtype=q_one_step.dtype, device=q_one_step.device)

    out = torch.eye(num_classes, dtype=q_one_step.dtype, device=q_one_step.device)
    for i in range(s + 1, t + 1):
        out = out @ q_one_step[i]
    return out


def q_sample_xt_given_x0(
    x0: torch.Tensor,
    t: torch.Tensor,
    q_cumulative: torch.Tensor,
    *,
    eps: float = 1e-6,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x0)."""
    logits = torch.log(_at(q_cumulative, t, x0) + eps)  # actually logprobs

    if noise is None:
        noise = torch.rand_like(logits)
    noise = noise.clamp(min=eps, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    # ref repo google-images q_sample and cloneofsimo q_sample both use Gumbel-Max.
    return torch.argmax(logits + gumbel_noise, dim=-1)


def q_posterior_xtm1_given_xt_x0(
    x0_or_x0_logits: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute posterior logits for q(x_{t-1}|x_t,x0_or_predx0)."""
    num_classes = q_cumulative.shape[-1]

    fact1 = _at(q_one_step_transposed, t, xt)

    t_minus_1 = (t - 1).clamp_min(0)
    if x0_or_x0_logits.dtype in (torch.int32, torch.int64):
        x0_logits = torch.log(F.one_hot(x0_or_x0_logits, num_classes).float() + eps)
        fact2 = _at(q_cumulative, t_minus_1, x0_or_x0_logits)
        t_zero_logits = x0_logits
    else:
        x0_logits = x0_or_x0_logits
        x0_probs = torch.softmax(x0_logits, dim=-1)
        fact2 = _at_onehot(q_cumulative, t_minus_1, x0_probs)
        t_zero_logits = x0_logits

    out = torch.log(fact1 + eps) + torch.log(fact2 + eps)

    # ref repo google-images uses t==0 branch (0-based indexing).
    # ref repo cloneofsimo uses t==1 branch due 1-based indexing; we standardize on 0-based.
    t_broadcast = t.view(xt.shape[0], *([1] * xt.ndim))
    return torch.where(t_broadcast == 0, t_zero_logits, out)


def q_posterior_xs_given_xt_x0(
    x0_or_x0_logits: torch.Tensor,
    xt: torch.Tensor,
    *,
    s: int,
    t: int,
    q_s_to_t_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Jump posterior used by sparse respacing: q(x_s | x_t, x0) for s < t.
    # Keep this separate from q_posterior_xtm1_given_xt_x0 to make reduced-step
    # sampling logic explicit and avoid mixing one-step and jump-step math.
    num_classes = q_cumulative.shape[-1]

    # q(x_t | x_s): gather column-conditioned probabilities using transpose.
    fact1 = q_s_to_t_transposed[xt]

    if x0_or_x0_logits.dtype in (torch.int32, torch.int64):
        x0_logits = torch.log(F.one_hot(x0_or_x0_logits, num_classes).float() + eps)
        fact2 = q_cumulative[s, x0_or_x0_logits, :]
        s_zero_logits = x0_logits
    else:
        x0_logits = x0_or_x0_logits
        x0_probs = torch.softmax(x0_logits, dim=-1)
        fact2 = torch.einsum("blk,kd->bld", x0_probs, q_cumulative[s])
        s_zero_logits = x0_logits

    out = torch.log(fact1 + eps) + torch.log(fact2 + eps)
    if s == 0:
        return s_zero_logits
    return out


def model_logits_to_p_logits(
    model_output: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    *,
    mode: str,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map model outputs to p_theta(x_{t-1}|x_t) logits."""
    if mode == "x0":
        pred_x0_logits = model_output

        # ref repo google-images p_logits maps x0 prediction through q_posterior.
        model_logits = q_posterior_xtm1_given_xt_x0(
            pred_x0_logits,
            xt,
            t,
            q_one_step_transposed,
            q_cumulative,
            eps=eps,
        )
        return model_logits, pred_x0_logits

    if mode == "x_tm1":
        # ref repo google-images has a NotImplemented path for x_tm1 in categorical case.
        # Divergence: we allow direct x_tm1 logits mode explicitly for experimentation.
        return model_output, model_output

    raise NotImplementedError(f"Unknown model prediction mode '{mode}'.")


def _categorical_kl_logits(
    logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float
) -> torch.Tensor:
    probs_p = torch.softmax(logits_p, dim=-1)
    log_probs_p = torch.log_softmax(logits_p + eps, dim=-1)
    log_probs_q = torch.log_softmax(logits_q + eps, dim=-1)
    return (probs_p * (log_probs_p - log_probs_q)).sum(dim=-1)


def d3pm_vlb_terms(
    *,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    model_prediction: str,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    eps: float = 1e-6,
    token_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute KL/NLL terms used by D3PM objectives."""
    true_logits = q_posterior_xtm1_given_xt_x0(  # used in tm1 term
        x0,
        xt,
        t,
        q_one_step_transposed,
        q_cumulative,
        eps=eps,
    )

    model_logits, pred_x0_logits = model_logits_to_p_logits(  # used in tm1 term
        model_output,
        xt,
        t,
        mode=model_prediction,
        q_one_step_transposed=q_one_step_transposed,
        q_cumulative=q_cumulative,
        eps=eps,
    )

    # ref repo google-images vb_terms_bpd computes KL plus t==0 decoder NLL branch.
    kl_token = _categorical_kl_logits(true_logits, model_logits, eps=eps)  # tm1 term
    decoder_nll_token = -torch.gather(  # t==0 term
        torch.log_softmax(model_logits, dim=-1),
        dim=-1,
        index=x0[..., None],
    ).squeeze(-1)

    # edge case: tm1 term only included when t > 0.
    t_broadcast = t.view(x0.shape[0], *([1] * (x0.ndim - 1)))
    vb_token = torch.where(t_broadcast == 0, decoder_nll_token, kl_token)

    if token_mask is None:
        token_weight = torch.ones_like(vb_token, dtype=torch.float32)
    else:
        token_weight = token_mask.to(vb_token.dtype)

    denom = token_weight.sum(dim=-1).clamp_min(1.0)
    vb_per_sample = (vb_token * token_weight).sum(dim=-1) / denom

    return {
        "vb_token": vb_token,
        "vb_per_sample": vb_per_sample,
        "kl_token": kl_token,
        "decoder_nll_token": decoder_nll_token,
        "model_logits": model_logits,
        "pred_x0_logits": pred_x0_logits,
    }


def d3pm_training_loss(
    *,
    backbone,
    x0: torch.Tensor,
    model_kwargs: Dict,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    num_steps: int,
    model_prediction: str,
    loss_type: str,
    hybrid_coeff: float,
    eps: float,
    token_mask: Optional[torch.Tensor],
    self_condition: bool,
    block_training: bool,
    time_conditioning: bool,
) -> Dict[str, torch.Tensor]:
    bsz, seq_len = x0.shape
    t = torch.randint(low=0, high=num_steps, size=(bsz,), device=x0.device)

    noise = torch.rand((bsz, seq_len, q_cumulative.shape[-1]), device=x0.device)
    xt = q_sample_xt_given_x0(x0, t, q_cumulative, eps=eps, noise=noise)
    if token_mask is not None:
        # ref repo insertdelete keeps sequence length/padding semantics explicit.
        # Divergence: we explicitly avoid corrupting padded positions.
        xt = torch.where(token_mask, xt, x0)

    xt_model = xt

    local_kwargs = dict(model_kwargs)
    t_norm = t.float() / max(num_steps - 1, 1)
    local_kwargs["timesteps"] = t_norm if time_conditioning else torch.zeros_like(t_norm)

    if self_condition:
        local_kwargs["self_conditions"] = torch.zeros(
            xt_model.shape[0],
            xt_model.shape[1],
            q_cumulative.shape[-1],
            device=xt_model.device,
        )
        if torch.rand(()) > 0.5:
            with torch.no_grad():
                sc_logits = backbone(xt_model, **local_kwargs)["out"]
            local_kwargs["self_conditions"] = sc_logits.detach()

    model_output = backbone(xt_model, **local_kwargs)["out"]
    if block_training:
        model_output = model_output[:, :seq_len]

    terms = d3pm_vlb_terms(
        x0=x0,
        xt=xt,
        t=t,
        model_output=model_output,
        model_prediction=model_prediction,
        q_one_step_transposed=q_one_step_transposed,
        q_cumulative=q_cumulative,
        eps=eps,
        token_mask=token_mask,
    )

    if token_mask is None:
        token_weight = torch.ones_like(terms["vb_token"], dtype=torch.float32)
    else:
        token_weight = token_mask.to(terms["vb_token"].dtype)
    denom = token_weight.sum(dim=-1).clamp_min(1.0)

    pred_x0_logits = terms["pred_x0_logits"]
    ce_token = F.cross_entropy(
        pred_x0_logits.transpose(-1, -2),
        x0,
        reduction="none",
    )
    ce_per_sample = (ce_token * token_weight).sum(dim=-1) / denom

    if loss_type == "kl":
        loss_per_sample = terms["vb_per_sample"]
    elif loss_type == "cross_entropy_x0":
        # ref repo google-images supports CE(x0) objective directly.
        loss_per_sample = ce_per_sample
    elif loss_type == "hybrid":
        # ref repo google-images/cloneofsimo both expose KL+lambda*CE hybrid style.
        loss_per_sample = terms["vb_per_sample"] + hybrid_coeff * ce_per_sample
    else:
        raise NotImplementedError(f"Unknown loss type '{loss_type}'.")

    return {
        "loss_per_sample": loss_per_sample,
        "vb_per_sample": terms["vb_per_sample"],
        "ce_per_sample": ce_per_sample,
        "pred_x0_logits": pred_x0_logits,
        "xt": xt,
        "t": t,
    }


def p_sample_step_d3pm(
    *,
    backbone,
    xt: torch.Tensor,
    t: torch.Tensor,
    s: int,
    model_kwargs: Dict,
    q_one_step: torch.Tensor,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    num_steps: int,
    model_prediction: str,
    eps: float,
    self_condition: bool,
    time_conditioning: bool,
    top: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    local_kwargs = dict(model_kwargs)
    t_norm = t.float() / max(num_steps - 1, 1)
    local_kwargs["timesteps"] = t_norm if time_conditioning else torch.zeros_like(t_norm)

    if self_condition and "self_conditions" not in local_kwargs:
        local_kwargs["self_conditions"] = torch.zeros(
            xt.shape[0], xt.shape[1], q_cumulative.shape[-1], device=xt.device
        )

    model_output = backbone(xt, **local_kwargs)["out"]
    if model_prediction == "x0":
        # ref repo google-images/cloneofsimo use one-step posterior in ancestral
        # full schedules. Divergence: sparse respacing here uses jump posterior
        # q(x_s | x_t, x0) for mathematically consistent coarse sampling.
        if not torch.all(t == t[0]):
            raise ValueError("Sampling expects a single timestep across the batch.")
        t_scalar = int(t[0].item())
        q_s_to_t = _q_transition_s_to_t(q_one_step, s=s, t=t_scalar)
        model_logits = q_posterior_xs_given_xt_x0(
            model_output,
            xt,
            s=s,
            t=t_scalar,
            q_s_to_t_transposed=q_s_to_t.transpose(0, 1),
            q_cumulative=q_cumulative,
            eps=eps,
        )
        pred_x0_logits = model_output
    else:
        model_logits, pred_x0_logits = model_logits_to_p_logits(
            model_output,
            xt,
            t,
            mode=model_prediction,
            q_one_step_transposed=q_one_step_transposed,
            q_cumulative=q_cumulative,
            eps=eps,
        )

    noise = torch.rand_like(model_logits).clamp(min=eps, max=1.0)
    gumbel_noise = -torch.log(-torch.log(noise))

    t_nonzero = (t != 0).float().view(xt.shape[0], *([1] * xt.ndim))
    # wierd heuristic: sample only from top probs
    if top is not None and int(top) > 1:
        k = min(int(top), model_logits.shape[-1])
        top_logits, top_idx = torch.topk(model_logits, k=k, dim=-1)
        top_noise = torch.rand_like(top_logits).clamp(min=eps, max=1.0)
        top_gumbel = -torch.log(-torch.log(top_noise))
        sampled_local = torch.argmax(top_logits + t_nonzero * top_gumbel, dim=-1)
        x_prev = top_idx.gather(-1, sampled_local[..., None]).squeeze(-1)
    else:
        x_prev = torch.argmax(model_logits + t_nonzero * gumbel_noise, dim=-1)

    # ref repo google-images p_sample sets no stochasticity at t==0.
    # ref repo cloneofsimo uses 1-based convention and masks noise when t==1.
    x_prev = torch.where(t[:, None] == 0, torch.argmax(model_logits, dim=-1), x_prev)

    return {
        "x_prev": x_prev,
        "model_logits": model_logits,
        "pred_x0_logits": pred_x0_logits,
    }


def sample_loop_d3pm(
    *,
    backbone,
    model_kwargs: Dict,
    q_one_step: torch.Tensor,
    q_one_step_transposed: torch.Tensor,
    q_cumulative: torch.Tensor,
    num_steps: int,
    model_prediction: str,
    eps: float,
    self_condition: bool,
    time_conditioning: bool,
    seq_len: int,
    batch_size: int,
    top: Optional[int],
    save_x: bool,
    save_p: bool,
    progress: bool,
    prior_type: str,
    prior_absorbing_idx: int,
    x_init: Optional[torch.Tensor] = None,
    steps_override: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    num_classes = q_cumulative.shape[-1]

    if x_init is not None:
        x = x_init.to(q_cumulative.device)
    elif prior_type == "absorbing":
        x = torch.full(
            (batch_size, seq_len),
            prior_absorbing_idx,
            dtype=torch.long,
            device=q_cumulative.device,
        )
    else:
        # ref repo google-images uses uniform prior for gaussian/uniform transitions.
        # Divergence from current denovo MDLM: no mask-prior here.
        x = torch.randint(
            low=0,
            high=num_classes,
            size=(batch_size, seq_len),
            device=q_cumulative.device,
        )

    full_schedule = list(range(num_steps - 1, -1, -1))
    if (
        steps_override is not None
        and steps_override > 0
        and steps_override != num_steps
    ):
        # ref repo google-images supports fewer eval steps via num_timesteps, but
        # does a prefix truncation of the reverse chain (no sparse respacing).
        # Divergence: we allow sparse linspace subsampling of reverse steps.
        idx = (
            torch.linspace(0, len(full_schedule) - 1, steps=steps_override)
            .round()
            .long()
            .tolist()
        )
        timesteps = [full_schedule[i] for i in idx]
    else:
        timesteps = full_schedule

    if save_x:
        x_save = torch.zeros(
            (len(timesteps) + 1, batch_size, seq_len),
            dtype=torch.int64,
            device=x.device,
        )
        x_save[0] = x
    if save_p:
        p_save = torch.zeros(
            (len(timesteps), batch_size, seq_len, num_classes),
            dtype=torch.float32,
            device=x.device,
        )

    final_logits = None

    pbar = tqdm(range(len(timesteps))) if progress else range(len(timesteps))
    for i in pbar:
        t_scalar = timesteps[i]
        t = torch.full((batch_size,), t_scalar, dtype=torch.long, device=x.device)
        s_scalar = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        step_out = p_sample_step_d3pm(
            backbone=backbone,
            xt=x,
            t=t,
            s=s_scalar,
            model_kwargs=model_kwargs,
            q_one_step=q_one_step,
            q_one_step_transposed=q_one_step_transposed,
            q_cumulative=q_cumulative,
            num_steps=num_steps,
            model_prediction=model_prediction,
            eps=eps,
            self_condition=self_condition,
            time_conditioning=time_conditioning,
            top=top,
        )
        x = step_out["x_prev"]
        final_logits = step_out["pred_x0_logits"]

        if self_condition:
            model_kwargs["self_conditions"] = step_out["model_logits"].detach()

        if save_x:
            x_save[i + 1] = x
        if save_p:
            p_save[i] = torch.softmax(step_out["pred_x0_logits"], dim=-1)

    output = {
        "prediction": x,
        "logits": final_logits,
    }
    if save_x:
        output["x_save"] = x_save.transpose(0, 1)
    if save_p:
        output["p_save"] = p_save.transpose(0, 1)
    return output


class D3PMDiffusion:
    def __init__(self, config: Dict, dictionary: Dict, backbone) -> None:
        self.config = config
        self.backbone = backbone
        self.vocab_size = len(dictionary)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps = float(config.get("eps", 1e-6))
        self.model_prediction = config.get("model_prediction", "x0")
        self.loss_type = config.get("loss_type", "hybrid")
        self.hybrid_coeff = float(config.get("hybrid_coeff", 1e-3))
        self.num_steps = int(config["T"])
        self.time_conditioning = bool(config.get("time_conditioning", True))

        transition_cfg = dict(config.get("transition", {}))
        if "type" not in transition_cfg:
            transition_cfg["type"] = "uniform"
        mats = build_q_matrices(
            num_classes=self.vocab_size,
            num_steps=self.num_steps,
            transition_cfg=transition_cfg,
        )
        self.q_one_step = mats["q_one_step"].float()
        self.q_cumulative = mats["q_cumulative"].float()
        self.q_one_step_transposed = mats["q_one_step_transposed"].float()
        self.betas = mats["betas"].float()

        sampling_cfg = config.get("sampling", {})
        self.prior_type = sampling_cfg.get("prior_type", "uniform")
        self.default_sampling_steps = int(sampling_cfg.get("steps", self.num_steps))
        self.prior_absorbing_idx = int(
            sampling_cfg.get(
                "prior_absorbing_idx", dictionary.get("<MASK>", self.vocab_size // 2)
            )
        )

    def _q_on_device(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.q_one_step_transposed.to(device),
            self.q_cumulative.to(device),
            self.q_one_step.to(device),
        )

    def d3pm_training_loss(
        self,
        backbone,
        x0: torch.Tensor,
        model_kwargs: Dict,
        block_training: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ):
        q_t_T, q_bar, _ = self._q_on_device(x0.device)
        return d3pm_training_loss(
            backbone=backbone,
            x0=x0,
            model_kwargs=model_kwargs,
            q_one_step_transposed=q_t_T,
            q_cumulative=q_bar,
            num_steps=self.num_steps,
            model_prediction=self.model_prediction,
            loss_type=self.loss_type,
            hybrid_coeff=self.hybrid_coeff,
            eps=self.eps,
            token_mask=token_mask,
            self_condition=bool(
                self.config.get("model", {}).get("self_condition", True)
            ),
            block_training=block_training,
            time_conditioning=self.time_conditioning,
        )

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
        del eps
        q_t_T, q_bar, q_t = self._q_on_device(self.device)
        sample_steps = (
            self.default_sampling_steps if num_steps is None else int(num_steps)
        )

        if x is None:
            batch_size = (
                len(model_kwargs["charge"])
                if model_kwargs.get("charge") is not None
                else model_kwargs["kv_features"].shape[0]
            )
            seq_len = int(self.config["model"]["length"])
        else:
            batch_size, seq_len = x.shape

        local_kwargs = dict(model_kwargs)
        if bool(self.config.get("model", {}).get("self_condition", True)):
            local_kwargs["self_conditions"] = torch.zeros(
                batch_size,
                seq_len,
                self.vocab_size,
                device=self.device,
            )

        return sample_loop_d3pm(
            backbone=self.backbone,
            model_kwargs=local_kwargs,
            q_one_step=q_t,
            q_one_step_transposed=q_t_T,
            q_cumulative=q_bar,
            num_steps=self.num_steps,
            model_prediction=self.model_prediction,
            eps=self.eps,
            self_condition=bool(
                self.config.get("model", {}).get("self_condition", True)
            ),
            time_conditioning=self.time_conditioning,
            seq_len=seq_len,
            batch_size=batch_size,
            top=top,
            save_x=save_x,
            save_p=save_p,
            progress=progress,
            prior_type=self.prior_type,
            prior_absorbing_idx=self.prior_absorbing_idx,
            x_init=x,
            steps_override=sample_steps,
        )
