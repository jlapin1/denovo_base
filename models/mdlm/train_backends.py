import torch.nn as nn
from torch.nn import functional as F


class MDLMTrainWrapperBase(nn.Module):
    """Base adapter from runner batches to diffusion-specific training losses."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, target, training_mask, block_decoding, custom_loss):
        """Compute per-sample or scalar losses for the selected diffusion backend."""
        raise NotImplementedError


class LegacyMDLMTrainWrapper(MDLMTrainWrapperBase):
    """Wrapper for legacy absorbing-mask MDLM objective."""

    def forward(self, batch, target, training_mask, block_decoding, custom_loss):
        embedding = self.model.encoder_embedding(batch)
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding['emb'],
            'seqmask': training_mask,
            'doubled': True if block_decoding else False,
        }
        backbone = self.model.decoder
        if custom_loss:
            model_output, weights, masked_token_mask, _ = self.model.diff_obj._forward_pass_diffusion(
                backbone, target, model_kwargs, block_decoding
            )
            loss = F.cross_entropy(model_output.transpose(-1, -2), target, reduction='none')
            weights = (target != self.model.decoder.NT).float() + 0.01 * (target == self.model.decoder.NT).float()
            loss = (weights * loss)[masked_token_mask]
        else:
            loss = self.model.diff_obj._forward_pass_diffusion(backbone, target, model_kwargs, block_decoding)
        return loss


class D3PMTrainWrapper(MDLMTrainWrapperBase):
    """Wrapper for general D3PM objective."""

    def forward(self, batch, target, training_mask, block_decoding, custom_loss):
        del custom_loss
        embedding = self.model.encoder_embedding(batch)
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding['emb'],
            'seqmask': training_mask,
            'doubled': True if block_decoding else False,
        }
        token_mask = (target != self.model.decoder.NT)
        backbone = self.model.decoder
        losses = self.model.diff_obj.d3pm_training_loss(
            backbone=backbone,
            x0=target,
            model_kwargs=model_kwargs,
            block_training=block_decoding,
            token_mask=token_mask,
        )
        return losses['loss_per_sample']


class InsertDeleteTrainWrapper(MDLMTrainWrapperBase):
    """Wrapper for Google-style insert/delete diffusion objective."""

    def forward(self, batch, target, training_mask, block_decoding, custom_loss):
        del training_mask, block_decoding, custom_loss
        embedding = self.model.encoder_embedding(batch)
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding['emb'],
            'seqmask': None,
            'doubled': False,
        }
        token_mask = (target != self.model.decoder.NT)
        backbone = self.model.decoder
        losses = self.model.diff_obj.insertdelete_training_loss(
            backbone=backbone,
            x0=target,
            model_kwargs=model_kwargs,
            token_mask=token_mask,
        )
        return losses['loss']


def create_mdlm_train_wrapper(model, backend='legacy_mdlm_mask'):
    """Factory returning the appropriate training wrapper for a backend name."""
    if backend in [None, 'legacy', 'legacy_mdlm_mask']:
        return LegacyMDLMTrainWrapper(model)
    if backend in ['d3pm_general', 'd3pm']:
        return D3PMTrainWrapper(model)
    if backend in ['insertdelete', 'insert_delete']:
        return InsertDeleteTrainWrapper(model)
    raise NotImplementedError(
        f"Unknown MDLM training backend '{backend}'. "
        "Expected one of: legacy_mdlm_mask, d3pm_general, insertdelete."
    )
