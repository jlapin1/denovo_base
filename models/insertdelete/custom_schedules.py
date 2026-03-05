from __future__ import annotations

import dataclasses
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from google_d3pm_ins_del import schedules as gschedules
from google_d3pm_ins_del import transition_operator as gtransition


def _make_pseudogaussian_standard_normal(beta: float, dim: int) -> np.ndarray:
    """Build a discretized Gaussian transition matrix.

    Args:
        beta: Noise level for this step.
        dim: Number of non-special vocabulary states.

    Returns:
        Row-stochastic transition matrix with shape `[dim, dim]`.
    """
    bounds = np.linspace(-2, 2, dim + 1)
    xs = (bounds[1:] + bounds[:-1]) / 2
    mu = np.sqrt(1 - beta) * xs
    distn = scipy.stats.norm(mu[:, None], np.sqrt(beta))
    cdfvals = distn.cdf(bounds)
    res = cdfvals[:, 1:] - cdfvals[:, :-1]
    return res / np.sum(res, axis=-1, keepdims=True)


def _gaussian_schedule(t1: float, t2: float, dim: int, num_steps: int) -> float:
    """Map continuous times `(t1, t2)` to a pseudo-Gaussian beta value.

    Args:
        t1: Start time in `[0, 1]`.
        t2: End time in `[0, 1]`, with `t2 >= t1`.
        dim: Number of non-special states.
        num_steps: Total number of diffusion steps.

    Returns:
        Per-step beta scalar clipped below 1.
    """
    min_standard_dev = 4 / dim
    min_beta = np.square(min_standard_dev)

    def remap(t):
        return -np.expm1(
            scipy.interpolate.interp1d(
                [1 / num_steps, 1],
                [np.log(min_beta), np.log(1)],
                fill_value="extrapolate",
            )(t)
        )

    v1 = remap(t1)
    v2 = remap(t2)
    return float(np.minimum(1 - v2 / v1, 0.999))


def _apply_insert_denylist(
    logits: jax.Array, token_denylist: Sequence[int]
) -> jax.Array:
    """Mask insertion logits for denied token ids.

    Args:
        logits: Logits/log-probs over modeled vocabulary.
        token_denylist: Token indices that inserted INS should not emit.

    Returns:
        Renormalized log-probabilities with denied ids set to zero mass.
    """
    if not token_denylist:
        return logits
    deny_arr = jnp.asarray(sorted(set(token_denylist)), dtype=jnp.int32)
    masked_logits = logits.at[deny_arr].set(-jnp.inf)
    return jax.nn.log_softmax(masked_logits)


def _build_interpolated_schedule(
    *,
    num_steps: int,
    vocab_size: int,
    transition_type: str,
    insert_type: str,
    final_relative_size: float,
    final_refresh_prob: float,
    initial_slope: float,
    acceleration: float,
    token_denylist: Sequence[int],
    mask_token: int | None,
):
    """Build a schedule mirroring Google `training_setup.state_init_fn`.

    Args:
        num_steps: Number of diffusion steps.
        vocab_size: Modeled vocabulary size on JAX side.
        transition_type: Transition operator family (`uniform`, `mask`, ...).
        insert_type: Insertion distribution family (`uniform` or `mask`).
        final_relative_size: Target expected length ratio near terminal time.
        final_refresh_prob: Terminal insertion refresh strength.
        initial_slope: Interpolator slope at early timesteps.
        acceleration: Interpolator curvature exponent.
        token_denylist: Tokens disallowed for insertion outputs.
        mask_token: Optional modeled id used by mask-based schedules.

    Returns:
        Google schedule object with per-step and cumulative one-step distributions.
    """
    if transition_type == "text8_nn":
        if vocab_size != 27:
            raise ValueError(
                "transition_type='text8_nn' requires vocab_size == 27."
            )
        transition_rate = 10.0
        vowels = "aeiouy"
        space_ix = 26
        vowel_ixs = np.array([ord(x) - ord("a") for x in vowels])
        consonant_ixs = np.array(
            [x for x in range(27) if x not in vowel_ixs and x != space_ix]
        )
        rate_matrix = np.full([27, 27], 0.5 * 1 / 27)
        rate_matrix[vowel_ixs[:, None], vowel_ixs[None, :]] += (
            0.5 * 1 / len(vowel_ixs)
        )
        rate_matrix[consonant_ixs[:, None], consonant_ixs[None, :]] += (
            0.5 * 1 / len(consonant_ixs)
        )
        rate_matrix = rate_matrix - np.diagflat(np.sum(rate_matrix, axis=0))

        def transition_distn_fn(t1, t2):
            return gtransition.MatrixOperator(
                scipy.linalg.expm(
                    transition_rate * (t2**1.8 - t1**1.8) * rate_matrix
                )
            )

    elif transition_type == "mask":
        if mask_token is None:
            raise ValueError(
                "transition_type='mask' requires a valid mask token in modeled vocab."
            )

        def transition_distn_fn(t1, t2):
            diagonal_value = (1 - t2) / (1 - t1)
            return gtransition.MaskDiffusionOperator(
                vocab_size, int(mask_token), jnp.log(diagonal_value)
            )

    elif transition_type == "delayed_mask":
        if mask_token is None:
            raise ValueError(
                "transition_type='delayed_mask' requires a valid mask token in modeled vocab."
            )

        def transition_distn_fn(t1, t2):
            diagonal_value = (1 - jnp.maximum(0.0, 2 * t2 - 1)) / (
                1 - jnp.maximum(0.0, 2 * t1 - 1)
            )
            return gtransition.MaskDiffusionOperator(
                vocab_size, int(mask_token), jnp.log(diagonal_value)
            )

    elif transition_type == "uniform":

        def transition_distn_fn(t1, t2):
            diagonal_value = (1 - t2) / (1 - t1)
            return gtransition.UniformDiffusionOperator(
                vocab_size, jnp.log(diagonal_value)
            )

    elif transition_type == "pseudogaussian":
        if vocab_size <= 2:
            raise ValueError(
                "transition_type='pseudogaussian' requires vocab_size > 2."
            )

        def transition_distn_fn(t1, t2):
            beta = _gaussian_schedule(t1, t2, vocab_size - 2, num_steps)
            mat = _make_pseudogaussian_standard_normal(beta, vocab_size - 2)
            mat = np.pad(mat, [(2, 0), (2, 0)], "constant")
            mat[0, 0] = 1
            mat[1, 1] = 1
            return gtransition.MatrixOperator(mat)

    else:
        raise NotImplementedError(f"Unknown insert/delete transition_type '{transition_type}'.")

    def insertion_distn_fn(t):
        if insert_type == "uniform":
            # ref repo google_d3pm_ins_del/training_setup.py modulates insertion
            # logits by transition_distn_fn(0, t) so inserts match corruption level.
            logits = jax.nn.log_softmax(jnp.zeros([vocab_size]))
            logits = _apply_insert_denylist(logits, token_denylist)
            logits = transition_distn_fn(0, t).apply(logits, is_distn=True, log=True)
            return _apply_insert_denylist(logits, token_denylist)
        if insert_type == "mask":
            if mask_token is None:
                raise ValueError(
                    "insert_type='mask' requires a valid mask token in modeled vocab."
                )
            logits = jnp.log(jax.nn.one_hot(int(mask_token), vocab_size))
            return _apply_insert_denylist(logits, token_denylist)
        raise NotImplementedError(f"Unknown insert_type '{insert_type}'.")

    def relative_size_fn(t):
        return final_relative_size + (1.0 - final_relative_size) * (1.0 - t)

    def refresh_prob_fn(t):
        return final_refresh_prob * t

    def interpolator(u):
        return initial_slope * u + (1 - initial_slope) * u**acceleration

    return gschedules.schedule_from_interpolators(
        num_steps=num_steps,
        transition_distn_fn=transition_distn_fn,
        insertion_distn_fn=insertion_distn_fn,
        relative_size_fn=relative_size_fn,
        refresh_prob_fn=refresh_prob_fn,
        interpolator=interpolator,
    )


def precompute_delete_count_marginals(schedule, max_len: int):
    """Attach delete-count marginal tables to cumulative distributions.

    Args:
        schedule: Google insert/delete schedule.
        max_len: Max sequence length used by the Torch wrapper.

    Returns:
        Schedule with precomputed delete-count marginals for faster DP lookups.
    """
    precomputed_cumulative = jax.lax.map(
        lambda distn: distn.with_precomputed_delete_count_marginals(max_len),
        schedule.cumulative,
    )
    return dataclasses.replace(schedule, cumulative=precomputed_cumulative)


def build_schedule_from_config(
    *,
    num_steps: int,
    vocab_size: int,
    schedule_cfg: Dict,
    token_denylist: Sequence[int] = (),
    mask_token: int | None = None,
    max_len: int | None = None,
):
    """Build insert/delete schedule from wrapper config.

    This wrapper intentionally mirrors Google `training_setup.state_init_fn`
    schedule construction via `schedule_from_interpolators`.

    Args:
        num_steps: Number of diffusion steps.
        vocab_size: Modeled vocabulary size on JAX side.
        schedule_cfg: User config dict controlling schedule family and params.
        token_denylist: JAX token ids disallowed for insertion outputs.
        mask_token: Optional JAX token id for mask schedule variants.
        max_len: If set, precompute delete-count marginals up to this length.

    Returns:
        Google insert/delete schedule object used for target construction/sampling.
    """
    unique_deny = tuple(sorted(set(token_denylist)))
    if len(unique_deny) >= vocab_size:
        raise ValueError(
            "insert/delete token_denylist excludes all modeled vocabulary tokens."
        )

    transition_type = str(schedule_cfg.get("transition_type", "uniform"))
    insert_type = str(schedule_cfg.get("insert_type", "uniform"))
    final_relative_size = float(schedule_cfg.get("final_relative_size", 1.0))
    final_refresh_prob = float(schedule_cfg.get("final_refresh_prob", 0.15))
    initial_slope = float(schedule_cfg.get("initial_slope", 1.0))
    acceleration = float(schedule_cfg.get("acceleration", 8.0))
    schedule = _build_interpolated_schedule(
        num_steps=num_steps,
        vocab_size=vocab_size,
        transition_type=transition_type,
        insert_type=insert_type,
        final_relative_size=final_relative_size,
        final_refresh_prob=final_refresh_prob,
        initial_slope=initial_slope,
        acceleration=acceleration,
        token_denylist=unique_deny,
        mask_token=mask_token,
    )

    if max_len is not None and int(max_len) > 0:
        schedule = precompute_delete_count_marginals(schedule, int(max_len))

    return schedule
