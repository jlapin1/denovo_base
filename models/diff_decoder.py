from copy import deepcopy
import numpy as np
from utils import Scale
import models.model_parts as mp
import torch as th
from torch import nn
I = nn.init
# beam search dependencies
import collections
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import heapq
from models.diffusion.gaussian_diffusion import _extract_into_tensor

def init_decoder_weights(module):
    if hasattr(module, 'first'):
        module.first.weight = I.xavier_uniform_(module.first.weight)
        if module.first.bias is not None:
            module.first.bias = I.zeros_(module.first.bias)
    if isinstance(module, (mp.SelfAttention, mp.CrossAttention)):
        module.Wo.weight = I.normal_(module.Wo.weight, 0.0, (1/3)*(module.h*module.d)**-0.5)
        if hasattr(module, 'qkv'):
            module.qkv.weight = I.normal_(module.qkv.weight, 0.0, (2/3)*module.indim**-0.5)
        if hasattr(module, 'Wb'):
            module.Wb.weight = I.zeros_(module.Wb.weight)
            module.Wb.bias = I.zeros_(module.Wb.bias)
        elif hasattr(module, 'Wpw'):
            module.Wpw.weight = I.zeros_(module.Wpw.weight)
            module.Wpw.bias = I.zeros_(module.Wpw.bias)
        if hasattr(module, 'Wg'):
            module.Wg.weight = I.zeros_(module.Wg.weight)
            module.Wg.bias = I.constant_(module.Wg.bias, 1.) # gate mostly open ~ 0.73
    elif isinstance(module, mp.FFN):
        module.W1.weight = I.xavier_uniform_(module.W1.weight)
        module.W1.bias = I.zeros_(module.W1.bias)
        module.W2.weight = I.normal_(module.W2.weight, 0.0, (1/3)*(module.indim*module.mult)**-0.5)
        #module.W2.weight = I.xavier_uniform_(module.W2.weight)
    elif isinstance(module, mp.TransBlock):
        if hasattr(module, 'embed') and module.embed_type == 'normembed':
            module.embed.weight = I.zeros_(module.embed.weight)
            module.embed.bias = I.zeros_(module.embed.bias)
    elif isinstance(module, nn.Linear):
        module.weight = I.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias = I.zeros_(module.bias)

def get_max_dic_value(dictionary, plus=1):
    # Must have at least as many embeddings as maximum integer
    return np.max(list(dictionary.values())) + plus

class base_diffusion_decoder(nn.Module):
    def __init__(self, 
        token_dict,
        running_units=512,
        d=64,
        h=8,
        ffn_multiplier=4,
        dropout=0,
        alphabet=False,
        prenorm=False,
        embed_type=None,
        timestep_dimension=128,
        kv_input_dimension=128,
        depth=6,
        use_charge=True,
        use_mass=True,
        use_leftover=True,
        precursor_dimension=128,
    ):
        super(base_diffusion_decoder, self).__init__()
        
        #####################
        # Output dictionary #
        #####################
        self.outdict = deepcopy(token_dict)
        self.NT = self.outdict['X']
        self.outdict['<EOS>'] = int(get_max_dic_value(self.outdict))
        self.EOS = self.outdict['<EOS>']
        
        self.rev_outdict = {n:m for m,n in self.outdict.items()}
        self.predcats = get_max_dic_value(self.outdict)
        self.scale = Scale(self.outdict)

        self.use_mass = use_mass
        self.use_leftover = use_leftover
        self.use_charge = use_charge
                
        self.precursor_dimension = precursor_dimension
        
        ############
        # Position #
        ############
        pos = mp.FourierFeatures(
            th.arange(100, dtype=th.float32), 1, 1000, running_units,
        )
        self.pos = nn.Parameter(pos, requires_grad=False)
        self.pos_modulator = nn.Parameter(th.tensor(0.1), requires_grad=True)
        
        ##############
        # Precursors #
        ##############
        self.use_charge = use_charge
        self.use_mass = use_mass
        self.atleast1 = True if (use_charge or use_mass or use_leftover) else False
        if self.atleast1:
            self.added_tokens = 1
            num = sum([use_charge, use_mass, use_leftover])
            if use_charge:
                #charge_embedder = nn.Embedding(8, embedding_dimension)
                self.charge_features = lambda charge: (
                    mp.FourierFeatures(charge, 1, 10, precursor_dimension)
                )
            if use_mass | use_leftover:
                self.mass_features = lambda mass: (
                    mp.FourierFeatures(mass, 0.001, 10000, precursor_dimension)
                )
            self.precursor_emb = nn.Sequential(
                nn.Linear(precursor_dimension*num, running_units)
            )
        else:
            self.added_tokens = 0
        
        ######################
        # Transformer blocks #
        ######################
        attention_dict = {
            'indim': running_units, 
            'd': d, 
            'h': h,
            'dropout': dropout,
            'alphabet': alphabet,
        }
        ffn_dict = {
            'indim': running_units,
            'unit_multiplier': ffn_multiplier, 
            'dropout': dropout,
            'alphabet': alphabet,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type='layer', 
                prenorm=prenorm, 
                embed_type=embed_type,
                embed_indim=timestep_dimension,
                is_cross=True,
                kvindim=kv_input_dimension,
            ) 
            for _ in range(depth)
        ])

    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])

    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = (th.arange(bs, device=intseq.device), lengths)
        intseq[eos_inds] = self.EOS

        return intseq
    
    def AddPosEmbed(self, seq_emb):
        return seq_emb + self.pos_modulator * self.pos[: seq_emb.shape[1]].unsqueeze(0)

    def AddPrecursorToken(self, seq_emb, charge=None, energy=None, mass=None, seq=None):
        
        # Add position to sequence
        out = self.AddPosEmbed(seq_emb)

        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(th.float32)
                ce_emb.append(self.charge_features(charge))
            if self.use_mass:
                ce_emb.append(self.mass_features(mass))
            if self.use_leftover:
                mass_so_far = self.scale.intseq2mz(seq, charge)
                leftover_mass = mass - mass_so_far
                ce_emb.append(self.mass_features(leftover_mass))
            if len(ce_emb) > 1:
                ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.precursor_emb(ce_emb)
            
            out = th.cat([ce_emb[:,None], out], dim=1)
        
        return out
    
    def RemovePrecursorToken(self, inp):
        return inp[:, self.added_tokens :]

    def sequence_mask(self, seq):
        return seq != self.NT

    def Main(self, inp, kv_feats, embed=None, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out, 
                kv_feats=kv_feats, 
                embed_feats=embed, 
                spec_mask=spec_mask,
                seq_mask=seq_mask 
            )
            out = out['out']
        
        return out

class DenovoDiffusionDecoder(base_diffusion_decoder):
    def __init__(self,
        token_dict,
        dec_config,
        diff_obj,
        input_output_units=128,
        running_units=512,
        d=64,
        h=8,
        dropout=0,
        ffn_multiplier=4,
        depth=6,
        timestep_dimension=128,
        precursor_dimension=128,
        alphabet=False,
        use_charge=False,
        use_mass=False,
        prenorm=False,
        self_condition=True,
        output_sigma=False,
        clip_denoised=False,
        clamp_denoised=False,
        **kwargs
    ):
        super().__init__(
            token_dict,
            running_units=running_units,
            d=d,
            h=h,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            alphabet=alphabet,
            prenorm=prenorm,
            embed_type='preembed',
            depth=depth,
            timestep_dimension=timestep_dimension,
            kv_input_dimension=dec_config['kv_indim'],
            use_charge=use_charge,
            use_mass=use_mass,
            precursor_dimension=precursor_dimension,
        )
        self.create_inpdict(token_dict)
        
        self.diff_obj = diff_obj
        self.dec_config = dec_config
        self.RU = RU = running_units
        self.input_output_units = input_output_units
        self.use_mass = use_mass
        self.use_charge = use_charge
        self.max_sl = dec_config['sequence_length']# + 1
        self.final_down_proj = nn.Linear(RU, input_output_units)
        self.final_down_proj = nn.Sequential(
            nn.Linear(RU, RU),
            nn.ReLU(),
            nn.Linear(RU, input_output_units)
        )
        self.output_sigma = output_sigma
        if output_sigma:
            self.sigma_down_proj = nn.Sequential(
                nn.Linear(RU, input_output_units),
                nn.Identity()
            )
        self.self_condition = self_condition
        self.clip_denoised = clip_denoised
        self.clamp_denoised = clamp_denoised
        self.time_dimension = timestep_dimension
        self.precursor_dimension = precursor_dimension

        """
        Transforming the x input to input for transformer block
        Note: identity in original paper if no self_condition
        """
        # TODO include sigma guess if learned_sigma -> 3*input_output_units
        x_input_dim = 2*input_output_units if self_condition else input_output_units
        self.input_proj_dec = nn.Sequential(
            nn.Linear(x_input_dim, RU),
            nn.Tanh(),
            nn.Linear(RU, RU)
        )

        """
        # The mapping of tokens to embeddings, and reverse, embeddings
        # to logits will have a shared weight that is only trained by
        # forward process.
        """
        # seq_emb: forward
        self.seq_emb = nn.Embedding(
            self.total_num_tokens, input_output_units, padding_idx=self.NT
        )
        self.seq_emb.weight = I.normal_(self.seq_emb.weight, 0, 0.03)
        with th.no_grad(): 
            self.seq_emb.weight[self.NT] = th.zeros_like(self.seq_emb.weight[self.NT])
        # lm_head: backward
        self.lm_head = nn.Linear(input_output_units, len(self.outdict))
        with th.no_grad():
            self.lm_head.weight = self.seq_emb.weight

        """Timestep embedding"""
        self.time_embed = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension)
        )
    
    def create_inpdict(self, token_dict):
        self.total_num_tokens = get_max_dic_value(self.outdict)

    def get_embed(self, seq):
        return self.seq_emb(seq)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def concat_self_cond(self, x, self_cond):
        return th.cat([x, self_cond], dim=-1)

    def forward(self, 
                x,
                timesteps,
                kv_feats, 
                charge=None, 
                energy=None, 
                mass=None,
                seqlen=None, 
                specmask=None,
                self_conditions=None,
                **kwargs
    ):
        time_emb = self.time_embed(mp.FourierFeatures(timesteps, 1, 10000, self.time_dimension))
        if self_conditions is not None:
            x = self.concat_self_cond(x, self_conditions)
        emb = self.input_proj_dec(x)
        emb = self.AddPrecursorToken(emb, charge=charge, mass=mass)
        out = self.Main(
            emb, kv_feats=kv_feats, embed=time_emb, 
            spec_mask=specmask, seq_mask=None
        )
        out = self.RemovePrecursorToken(out)
        out = self.final_down_proj(out)
        out_dict = {'mean': out}
        if self.output_sigma:
            logvar_fraction = self.sigma_down_proj(out)
            out_dict['var'] = logvar_fraction
        return out_dict

    def predict_sequence(self, embedding, batch, save_xcur=False, save_xstart=False, cond_fn=None, progress=False):
        shape = (
            embedding['emb'].shape[0],
            self.max_sl,
            self.input_output_units,
        )
        model_kwargs = {
            'kv_feats': embedding['emb'],
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
        }
        
        # Create fully noised real data
        device = model_kwargs['kv_feats'].device
        noise = th.randn(*shape, device=device)

        output = self.diff_obj.my_p_sample_loop(
            self,
            shape,
            noise=noise,
            denoised_fn=self.clamp if self.clamp_denoised else None,
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
            save_xcur=save_xcur,
            save_xstart=save_xstart,
            cond_fn=cond_fn,
            progress=progress,
        )
        logits = self.get_logits(output['final']) # bs, sl, predcats
        final = logits.argmax(dim=-1)
        
        return_ = {'prediction': final, 'logits': logits}
        if save_xcur:
            return_['xcur'] = output['xcur_save']
        if save_xstart:
            return_['xstart'] = output['xstart_save']

        return return_

    def clamp(self, x_0, *args):
        embedding = self.lm_head.weight # 24, 512
        dist = (x_0[...,None,:] - embedding[None, None]).square().sum(-1) # bs, 31, 24
        input_ids = dist.argmin(-1)
        #input_ids = self.get_logits(x_0).argmax(-1) # bs, 31
        return self.get_embed(input_ids)

    def set_clamp(self, boolean: bool):
        self.clamp_denoised = boolean

class MDLMDecoder(base_diffusion_decoder):
    def __init__(self,
        token_dict,
        decoder_config,
        running_units=512,
        d=64,
        h=8,
        dropout=0,
        ffn_multiplier=4,
        depth=6,
        timestep_dimension=128,
        precursor_dimension=128,
        alphabet=False,
        use_charge=False,
        use_mass=False,
        use_leftover=False,
        prenorm=False,
        self_condition=True,
        output_sigma=False,
        clip_denoised=False,
        clamp_denoised=False,
        **kwargs
    ):
        super().__init__(
            token_dict,
            running_units=running_units,
            d=d,
            h=h,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            alphabet=alphabet,
            prenorm=prenorm,
            embed_type=None,
            depth=depth,
            timestep_dimension=timestep_dimension,
            kv_input_dimension=decoder_config['kv_indim'],
            use_charge=use_charge,
            use_mass=use_mass,
            use_leftover=use_leftover,
            precursor_dimension=precursor_dimension,
        )
        self.finish_dict()
        self.timestep_dimension = timestep_dimension
        self.self_condition = self_condition

        self.max_sl = decoder_config['sequence_length'] # + 1

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension),
        )
        
        #self.lm_head = nn.Embedding(self.total_num_input_tokens, 
        self.embed_sequence = nn.Embedding(self.predcats, running_units)
        if self_condition:
            self.embed_self_conditions = nn.Sequential(nn.Linear(self.predcats, running_units))
        
        self.proj_begin = nn.Sequential(
            nn.Linear(running_units, running_units),
            nn.LayerNorm(running_units),
            nn.ReLU(),
        )
        self.proj_end = nn.Sequential(
            nn.Linear(running_units, running_units),
            nn.LayerNorm(running_units),
            nn.ReLU(),
            nn.Linear(running_units, self.predcats),
        )

    def finish_dict(self):
        self.outdict['<MASK>'] = int(get_max_dic_value(self.outdict))
        self.MASK = self.outdict['<MASK>']
        self.rev_outdict = {n:m for m,n in self.outdict.items()}
        self.predcats = get_max_dic_value(self.outdict)
        self.scale = Scale(self.outdict)

    def forward(self,
        x,
        timesteps,
        kv_features,
        charge,
        mass,
        specmask=None,
        self_conditions=None,
    ):
        # Timestep
        time_emb = self.time_embed(mp.FourierFeatures(timesteps, 0.000001, 10, self.timestep_dimension))

        # Beginning
        seq_emb = self.embed_sequence(x)
        if self.self_condition:
            seq_emb += self.embed_self_conditions(self_conditions)
        emb = self.AddPrecursorToken(seq_emb, charge=charge, mass=mass, seq=x) # position added inside
        emb = self.proj_begin(emb)
        
        # Middle
        out = self.Main(
            emb, 
            kv_feats=kv_features,
            embed=time_emb,
            spec_mask=specmask,
            seq_mask=None,
        )

        # End
        out = self.proj_end(out)
        out = self.RemovePrecursorToken(out)

        return out
    
    def predict_sequence(self, embedding, batch, save_x=False, save_p=False, top=None, num_steps=None, progress=False):
        model_kwargs = {
            'kv_features': embedding,
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
        }
        out = self.diff_obj._sample(
            save_x=save_x, 
            save_p=save_p, 
            top=top,
            num_steps=num_steps,
            progress=progress, 
            model_kwargs=model_kwargs
        )
        return out
