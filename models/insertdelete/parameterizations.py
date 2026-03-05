from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from google_d3pm_ins_del import forward_process as gfp


class InsertDeleteX0Parameterizer:
    """Convert x0-style model outputs to reverse-step marginals.

    The Google insert/delete math parameterizes the model via approximate
    distributions over x0/edit summary and then derives reverse marginals with
    `apply_x0_parameterization`. This helper keeps that conversion isolated from
    the Torch training/sampling loops.
    """

    def __init__(self, schedule) -> None:
        """Build a batched JAX conversion function for the provided schedule.

        Args:
            schedule: Google insert/delete schedule with `distns_at_step`.
        """
        self.schedule = schedule
        self._build_jax_apply_fn()

    def _build_jax_apply_fn(self) -> None:
        """Compile a batch conversion from approximate x0 marginals to p(x_t|x_{t+1})."""
        schedule = self.schedule

        def _single_apply(
            approx_was_insert_log_prob: jax.Array,
            approx_previous_token_log_probs: jax.Array,
            approx_delete_log_probs: jax.Array,
            xtplus1_tokens: jax.Array,
            xtplus1_length: jax.Array,
            t: jax.Array,
        ):
            xtplus1 = gfp.DynamicLengthSentinelSequence(
                tokens=xtplus1_tokens,
                length=xtplus1_length,
            )
            d_0_to_t, d_t_to_tplus1 = schedule.distns_at_step(t)
            precomputed_Attplus1_xtplus1 = jax.vmap(
                functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True)
            )(xtplus1.tokens)

            approximate_x0_guess = gfp.ReverseProcessMarginalDistribution(
                was_missing=xtplus1.insert_sentinel_mask(),
                was_insert_log_prob=approx_was_insert_log_prob,
                previous_token_log_probs=approx_previous_token_log_probs,
                log_prob_number_of_preceding_deletes=approx_delete_log_probs,
                length=xtplus1_length,
            )

            adjusted = gfp.apply_x0_parameterization(
                approximate_x0_guess=approximate_x0_guess,
                xtplus1=xtplus1,
                d_0_to_t=d_0_to_t,
                d_t_to_tplus1=d_t_to_tplus1,
                precomputed_Attplus1_xtplus1=precomputed_Attplus1_xtplus1,
            )
            return (
                adjusted.was_insert_log_prob,
                adjusted.previous_token_log_probs,
                adjusted.log_prob_number_of_preceding_deletes,
            )

        self._jax_apply_batch = jax.jit(
            jax.vmap(
                _single_apply,
                in_axes=(0, 0, 0, 0, 0, 0),
            )
        )

    def apply_numpy(
        self,
        *,
        approx_was_insert_log_prob: np.ndarray,
        approx_previous_token_log_probs: np.ndarray,
        approx_delete_log_probs: np.ndarray,
        xtplus1_tokens: np.ndarray,
        xtplus1_length: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply x0 parameterization using NumPy inputs/outputs.

        Args:
            approx_was_insert_log_prob: Log-prob of "token was inserted", shape [B, L].
            approx_previous_token_log_probs: Conditional previous-token log-probs, shape [B, L, K].
            approx_delete_log_probs: Delete-count log-probs, shape [B, L+1, D].
            xtplus1_tokens: Current noised tokens in Google sentinel space, shape [B, L].
            xtplus1_length: Current noised sequence lengths, shape [B].
            t: Diffusion steps for each sample, shape [B].

        Returns:
            Tuple `(was_insert_log_prob, previous_token_log_probs, delete_log_probs)`
            with the same batch shapes as the corresponding inputs.
        """
        converted = self._jax_apply_batch(
            jnp.asarray(approx_was_insert_log_prob, dtype=jnp.float32),
            jnp.asarray(approx_previous_token_log_probs, dtype=jnp.float32),
            jnp.asarray(approx_delete_log_probs, dtype=jnp.float32),
            jnp.asarray(xtplus1_tokens, dtype=jnp.int32),
            jnp.asarray(xtplus1_length, dtype=jnp.int32),
            jnp.asarray(t, dtype=jnp.int32),
        )
        was_insert_log_prob, previous_token_log_probs, delete_log_probs = jax.device_get(
            converted
        )
        return (
            np.asarray(was_insert_log_prob, dtype=np.float32),
            np.asarray(previous_token_log_probs, dtype=np.float32),
            np.asarray(delete_log_probs, dtype=np.float32),
        )

