import torch.nn as nn
from torch.nn import functional as F


class MDLMTrainWrapperBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, target, training_mask, block_decoding, custom_loss):
        raise NotImplementedError


class LegacyMDLMTrainWrapper(MDLMTrainWrapperBase):
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


def create_mdlm_train_wrapper(model, backend='legacy_mdlm_mask'):
    if backend in [None, 'legacy', 'legacy_mdlm_mask']:
        return LegacyMDLMTrainWrapper(model)
    if backend in ['d3pm_general', 'd3pm']:
        return D3PMTrainWrapper(model)
    raise NotImplementedError(
        f"Unknown MDLM training backend '{backend}'. "
        "Expected one of: legacy_mdlm_mask, d3pm_general."
    )
