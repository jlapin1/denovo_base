from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def safe_exp_weighted_torch(
    log_gate: torch.Tensor, log_result: torch.Tensor
) -> torch.Tensor:
    """Torch translation of Google `math_util.safe_exp_weighted`.

    Args:
        log_gate: Log-space weights (can include `-inf`).
        log_result: Log-space values to weight.

    Returns:
        Elementwise `exp(log_gate) * log_result` with impossible gates masked to 0.
    """
    impossible = torch.isneginf(log_gate)
    safe_result = torch.where(impossible, log_result.detach(), log_result)
    if_possible = torch.where(
        torch.isneginf(safe_result),
        torch.full_like(safe_result, float("-inf")),
        torch.exp(log_gate) * safe_result,
    )
    return torch.where(impossible, torch.zeros_like(if_possible), if_possible)


@dataclass
class ReverseProcessMarginalDistributionTorch:
    """Torch analogue of Google `ReverseProcessMarginalDistribution`.

    This class mirrors `google_d3pm_ins_del.forward_process.ReverseProcessMarginalDistribution`
    and keeps the same field names so training code can read like Google's
    `sample_based_elbo_term_loss` implementation.
    """

    was_missing: torch.Tensor
    was_insert_log_prob: torch.Tensor
    previous_token_log_probs: torch.Tensor
    log_prob_number_of_preceding_deletes: torch.Tensor
    length: torch.Tensor

    @staticmethod
    def cross_entropy(
        *,
        samples_from: "ReverseProcessMarginalDistributionTorch",
        model_output: "ReverseProcessMarginalDistributionTorch",
        return_extra: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]] | torch.Tensor:
        """Torch translation of Google `ReverseProcessMarginalDistribution.cross_entropy`.

        Args:
            samples_from: Target marginal distribution from forward-process math.
            model_output: Model-predicted reverse marginals.
            return_extra: If True, also return individual delete/insert/token terms.

        Returns:
            Per-sample cross-entropy, and optionally its decomposed terms.
        """
        _, max_len = samples_from.was_insert_log_prob.shape
        arange_seq = torch.arange(max_len, device=samples_from.length.device)[None]
        valid_token_mask = (~samples_from.was_missing) & (
            arange_seq < samples_from.length[:, None]
        )
        valid_token_or_eos_mask = torch.zeros(
            (samples_from.length.shape[0], max_len + 1),
            dtype=torch.bool,
            device=samples_from.length.device,
        )
        valid_token_or_eos_mask[:, :max_len] = valid_token_mask
        valid_token_or_eos_mask.scatter_(
            1,
            samples_from.length[:, None].clamp_max(max_len),
            True,
        )

        delete_term = torch.where(
            valid_token_or_eos_mask[:, :, None],
            safe_exp_weighted_torch(
                samples_from.log_prob_number_of_preceding_deletes,
                model_output.log_prob_number_of_preceding_deletes,
            ),
            torch.zeros_like(model_output.log_prob_number_of_preceding_deletes),
        ).sum(dim=(1, 2))

        insert_term = torch.where(
            valid_token_mask,
            safe_exp_weighted_torch(
                samples_from.was_insert_log_prob,
                model_output.was_insert_log_prob,
            ),
            torch.zeros_like(model_output.was_insert_log_prob),
        ).sum(dim=1)

        token_term = torch.where(
            valid_token_mask[:, :, None],
            safe_exp_weighted_torch(
                samples_from.previous_token_log_probs,
                model_output.previous_token_log_probs,
            ),
            torch.zeros_like(model_output.previous_token_log_probs),
        ).sum(dim=(1, 2))

        total = delete_term + insert_term + token_term
        if return_extra:
            return total, {
                "delete_term": delete_term,
                "insert_term": insert_term,
                "token_term": token_term,
            }
        return total
