import torch as th
from torch import nn
import models.model_parts as mp
from models.diffusion.gaussian_diffusion import _extract_into_tensor
from models.diffusion.model_utils import create_diffusion
import os
from glob import glob
import yaml

device = th.device("cuda" if th.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(
        self,
        diff_dir,
        num_input_tokens=24,
        num_output_classes=8,
        running_units=512,
        d=64,
        h=8,
        dropout=0,
        embed_type='preembed',
        prenorm=False,
        ffn_multiplier=2,
        depth=6,
        timestep_dimension=128,
        null_token=22,
        eos_token=23,
    ):
        super(Classifier, self).__init__()
        self.timestep_dimension = timestep_dimension
        self.NT = null_token
        self.EOS = eos_token
        self.dir = diff_dir
        self.num_input_tokens = num_input_tokens
        self.running_units = running_units
        
        """Diffusion object"""
        self.configure_diffusion_object(diff_dir)

        """Timestep embedding"""
        self.time_embed = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension)
        )

        """Position embedding"""
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        self.pos = nn.Parameter(
            mp.FourierFeatures(th.arange(100), 1, 1000, running_units), 
            requires_grad=True
        )

        """seq_emb"""
        self.configure_seq_embed(diff_dir)
        
        """Transformer blocks"""
        attention_dict = {
            'indim': running_units, 
            'd': d, 
            'h': h,
            'dropout': dropout,
            'alphabet': False,
        }
        ffn_dict = {
            'indim': running_units,
            'unit_multiplier': ffn_multiplier, 
            'dropout': dropout,
            'alphabet': False,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type='layer',
                prenorm=prenorm, 
                embed_type=embed_type,
                embed_indim=timestep_dimension,
                is_cross=False,
                kvindim=None,
            ) 
            for _ in range(depth)
        ])

        """Final discriminator prediction"""
        self.final = nn.Sequential(
            nn.Linear(running_units, running_units, bias=False),
            nn.LayerNorm(running_units),
            nn.ReLU(),
            nn.Linear(running_units, num_output_classes),
        )
    
    def load_weights(self, ckpt):
        self.load_state_dict(th.load(ckpt, map_location=device))

    def configure_seq_embed(self, svdir):
        # Layer
        self.seq_emb = nn.Embedding(self.num_input_tokens, self.running_units, padding_idx=self.NT)
        
        if svdir is not None:
            # Locate the saved weight
            weights_path = glob(os.path.join(svdir, "weights/*high*wts"))[0]
            weight_dict = th.load(weights_path, map_location=device,)
            seq_emb_weight = weight_dict['decoder.seq_emb.weight']
            
            # Assign the weight
            assert self.seq_emb.weight.shape == seq_emb_weight.shape, seq_emb_weight.shape
            with th.no_grad():
                self.seq_emb.weight = nn.Parameter(seq_emb_weight)
            
            # Don't train the seq_emb
            self.seq_emb.weight.requires_grad = False

    def configure_diffusion_object(self, diff_dir):
        yaml_file = os.path.join(diff_dir, "yaml", "config.yaml")
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        diff_config = config["decoder_diff"]['diffusion_config']
        diff_config['pad_tok_id'] = self.NT
        diff_config['resume_checkpoint'] = False
        self.diff_obj = create_diffusion(**diff_config)
 
    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.EOS

        return intseq

    def get_x_start(self, intseq):
        intseq = self.append_null_token(intseq)
        lengths = (intseq != self.NT).sum(1)
        intseq = self.replace_with_eos_token(intseq, lengths)
        x_start_mean = self.seq_emb(intseq)
        std = _extract_into_tensor(
            self.diff_obj.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]).to(x_start_mean.device),
            x_start_mean.shape,
        )
        x_start = self.diff_obj.get_x_start(x_start_mean, std)
        return x_start

    def noisy_x(self, x_start, t):
        return self.diff_obj.q_sample(x_start, t, noise=None)
    
    def get_noisy_x(self, intseq, t):
        x_start = self.get_x_start(intseq)
        noisy_x = self.noisy_x(x_start, t)
        return noisy_x

    def Main(self, inp, time_embed, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out,
                embed_feats=time_embed,
                spec_mask=spec_mask,
                seq_mask=seq_mask
            )
            out = out['out']

        return out

    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])

    def forward(self, latent, timesteps):
        """
        Model is built to classify noisy latents
        - During training, peptide sequences are turned into x_start and forward
          diffused to random timesteps.
        - During guided diffusion, model accepts intermediate latents from
          diffusion model.
        """

        # Time embedding
        time_emb = self.time_embed(mp.FourierFeatures(timesteps, 1, 10000, self.timestep_dimension))

        # Process latent
        latent_ = latent + self.alpha * self.pos[:latent.shape[1]]
        out = self.Main(latent_, time_emb)
        
        # Logits
        out = self.final(out)

        return out.mean(1)

