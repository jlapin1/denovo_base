from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
import torch

from google_d3pm_ins_del import forward_process as gfp


class InsertDeleteTorchJaxBridge:
    """Bridge utilities for Torch token ids <-> Google JAX sentinel sequences."""

    def __init__(
        self,
        dictionary: Dict,
        *,
        max_len: int,
        include_mask_in_vocab: bool = False,
    ):
        """Build vocabulary maps and conversion helpers.

        Args:
            dictionary: Repo token->id dictionary.
            max_len: Maximum modeled sequence length for insert/delete process.
            include_mask_in_vocab: Whether `<MASK>` should be part of modeled
                JAX vocabulary (required for mask-based schedule variants).
        """
        self.max_len = int(max_len)
        self.pad_token_id = int(dictionary["X"])
        self.eos_token_id = int(dictionary.get("<EOS>", -1))
        self.sos_token_id = int(dictionary.get("<SOS>", -1))
        self.mask_token_id = int(dictionary.get("<MASK>", -1))
        self.insert_token_id = int(dictionary["<INS>"])
        self.delete_token_id = int(dictionary["<DEL>"])
        self.include_mask_in_vocab = bool(include_mask_in_vocab)

        excluded = {
            self.pad_token_id,
            self.sos_token_id,
            self.insert_token_id,
            self.delete_token_id,
        }
        if not self.include_mask_in_vocab and self.mask_token_id >= 0:
            excluded.add(self.mask_token_id)
        if self.eos_token_id >= 0:
            excluded.add(self.eos_token_id)
        # EOS is a virtual boundary for insert/delete math, not a modeled token.
        real_torch_ids = sorted({idx for idx in dictionary.values() if idx not in excluded})
        self.real_torch_ids_tensor = torch.tensor(real_torch_ids, dtype=torch.long)
        self.jax_to_torch = np.array(real_torch_ids, dtype=np.int64)
        self.real_vocab_size = int(self.jax_to_torch.shape[0])

        max_token_id = int(max(dictionary.values()))
        self.torch_to_jax_lut = np.full((max_token_id + 1,), -1, dtype=np.int32)
        self.torch_to_jax_lut[self.jax_to_torch] = np.arange(
            self.jax_to_torch.shape[0], dtype=np.int32
        )

    def resolve_deny_insert_jax(
        self, *, schedule_cfg: Dict, dictionary: Dict
    ) -> Tuple[int, ...]:
        """Map config deny-list token names to modeled JAX ids."""
        deny_insert_tokens = set(schedule_cfg.get("deny_insert_tokens", []))
        deny_insert_jax = []
        for token_name in deny_insert_tokens:
            if token_name not in dictionary:
                continue
            tok = int(dictionary[token_name])
            if not (0 <= tok < self.torch_to_jax_lut.shape[0]):
                continue
            jax_tok = int(self.torch_to_jax_lut[tok])
            if jax_tok >= 0:
                deny_insert_jax.append(jax_tok)
        return tuple(sorted(set(deny_insert_jax)))

    def mask_token_jax(self) -> int | None:
        """Return modeled JAX id for `<MASK>` token, if present."""
        if 0 <= self.mask_token_id < self.torch_to_jax_lut.shape[0]:
            candidate = int(self.torch_to_jax_lut[self.mask_token_id])
            if candidate >= 0:
                return candidate
        return None

    def torch_x0_to_jax_dynamic_sequence(
        self,
        x0: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> gfp.DynamicLengthSentinelSequence:
        """Convert clean Torch targets to Google DynamicLengthSentinelSequence."""
        bsz, seq_len = x0.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"InsertDeleteDiffusion expected sequence length <= {self.max_len}, got {seq_len}."
            )

        x0_np = x0.detach().cpu().numpy().astype(np.int64, copy=False)
        token_mask_np = token_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
        lengths = token_mask_np.sum(axis=1).astype(np.int32)
        valid_prefix_mask = np.arange(seq_len)[None, :] < lengths[:, None]

        if np.any(valid_prefix_mask & (x0_np == self.pad_token_id)):
            raise ValueError("Encountered PAD token in valid prefix; cannot diffuse PAD.")
        if np.any(valid_prefix_mask & (x0_np == self.insert_token_id)):
            raise ValueError("Encountered INS token in x0 prefix; expected clean target.")
        if np.any(valid_prefix_mask & (x0_np == self.delete_token_id)):
            raise ValueError("Encountered DEL token in x0 prefix; expected clean target.")
        if self.mask_token_id >= 0 and np.any(valid_prefix_mask & (x0_np == self.mask_token_id)):
            raise ValueError("Encountered MASK token in x0 prefix; expected clean target.")
        if self.eos_token_id >= 0 and np.any(valid_prefix_mask & (x0_np == self.eos_token_id)):
            raise ValueError(
                "Encountered EOS token in x0 prefix; EOS must stay a virtual boundary."
            )
        if np.any(x0_np < 0) or np.any(x0_np >= self.torch_to_jax_lut.shape[0]):
            raise ValueError("Encountered token id outside insert/delete vocabulary LUT.")

        mapped_prefix = self.torch_to_jax_lut[x0_np]
        invalid_vocab = valid_prefix_mask & (mapped_prefix < 0)
        if np.any(invalid_vocab):
            bad_tokens = np.unique(x0_np[invalid_vocab]).tolist()
            raise ValueError(f"Token ids {bad_tokens} are not in insert/delete modeled vocabulary.")

        mapped_tokens = np.full(
            (bsz, self.max_len),
            gfp.DynamicLengthSentinelSequence.ERROR,
            dtype=np.int32,
        )
        mapped_tokens[:, :seq_len] = np.where(
            valid_prefix_mask, mapped_prefix, gfp.DynamicLengthSentinelSequence.ERROR
        )
        return gfp.DynamicLengthSentinelSequence(
            tokens=jnp.asarray(mapped_tokens, dtype=jnp.int32),
            length=jnp.asarray(lengths, dtype=jnp.int32),
        )

    def jax_tokens_lengths_to_torch_batch(
        self,
        tokens: np.ndarray,
        lengths: np.ndarray,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert Google sentinel tokens to repo Torch token ids."""
        bsz = tokens.shape[0]
        lengths = np.clip(lengths.astype(np.int32), 0, self.max_len)
        active = np.arange(self.max_len)[None, :] < lengths[:, None]
        out = np.full((bsz, self.max_len), self.pad_token_id, dtype=np.int64)
        is_insert = active & (tokens == gfp.DynamicLengthSentinelSequence.INSERT_SENTINEL)
        is_delete = active & (tokens == gfp.DynamicLengthSentinelSequence.DELETE_SENTINEL)
        is_vocab = active & (tokens >= 0)
        if np.any(is_vocab & (tokens >= self.jax_to_torch.shape[0])):
            raise ValueError("Out-of-range Google vocab token produced by insert/delete process.")
        out[is_insert] = self.insert_token_id
        out[is_delete] = self.delete_token_id
        out[is_vocab] = self.jax_to_torch[tokens[is_vocab]]
        return torch.tensor(out, dtype=torch.long, device=device)

    def prepare_tokens_for_x0_parameterization(
        self,
        xtplus1_tokens: np.ndarray,
        xtplus1_length: np.ndarray,
    ) -> np.ndarray:
        """Replace padded ERROR slots with DEL for Google x0-parameterization."""
        prepared = np.array(xtplus1_tokens, dtype=np.int32, copy=True)
        max_len = prepared.shape[1]
        pad_mask = np.arange(max_len)[None, :] >= xtplus1_length[:, None]
        prepared[pad_mask] = gfp.DynamicLengthSentinelSequence.DELETE_SENTINEL
        return prepared

    def jax_targets_to_torch(
        self, batch_targets: Dict, *, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Convert batched JAX target dict to Torch tensors."""
        xt_tokens = np.asarray(batch_targets["xt_tokens"], dtype=np.int32)
        xt_lengths = np.asarray(batch_targets["xt_length"], dtype=np.int32)
        xt_torch = self.jax_tokens_lengths_to_torch_batch(
            xt_tokens, xt_lengths, device=device
        )

        return {
            "xt": xt_torch,
            "xt_tokens_jax": torch.tensor(xt_tokens, dtype=torch.int32, device=device),
            "xt_length": torch.tensor(xt_lengths, dtype=torch.int32, device=device),
            "t": torch.tensor(
                np.asarray(batch_targets["t"], dtype=np.int64),
                dtype=torch.long,
                device=device,
            ),
            "t_weight": torch.tensor(
                np.asarray(batch_targets["t_weight"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "expected_q": torch.tensor(
                np.asarray(batch_targets["expected_q"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "valid": torch.tensor(
                np.asarray(batch_targets["valid"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "target_was_missing": torch.tensor(
                np.asarray(batch_targets["target_was_missing"], dtype=np.bool_),
                dtype=torch.bool,
                device=device,
            ),
            "target_was_insert_logprob": torch.tensor(
                np.asarray(batch_targets["target_was_insert_logprob"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "target_prev_token_logprob": torch.tensor(
                np.asarray(batch_targets["target_prev_token_logprob"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "target_delete_logprob": torch.tensor(
                np.asarray(batch_targets["target_delete_logprob"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            "target_length": torch.tensor(
                np.asarray(batch_targets["target_length"], dtype=np.int64),
                dtype=torch.long,
                device=device,
            ),
            "target_length_logprob": torch.tensor(
                np.asarray(batch_targets["target_length_logprob"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
        }
