import torch as th
from torch import nn
import models.model_parts as mp
from models.diffusion.gaussian_diffusion import _extract_into_tensor
from models.diffusion.model_utils import create_diffusion
from models.seq2seq import Seq2SeqMDLM
import os
from glob import glob
import yaml
import utils

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


class Regressor4MDLM(nn.Module):
    def __init__(
        self,
        diff_dir,
        amod_dict,
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
        data_mean = 0.,
        data_std = 1.,
    ):
        super(Regressor4MDLM, self).__init__()
        self.timestep_dimension = timestep_dimension
        self.NT = null_token
        self.EOS = eos_token
        #self.dir = diff_dir
        self.num_input_tokens = len(amod_dict) + 1
        self.running_units = running_units
        
        self.data_mean = nn.Parameter(th.tensor(data_mean), requires_grad=False)
        self.data_std = nn.Parameter(th.tensor(data_std), requires_grad=False)
        
        """Diffusion model and object"""
        self.configure_diffusion(diff_dir, amod_dict) 

        """Timestep embedding"""
        #self.time_embed = nn.Sequential(
        #    nn.Linear(timestep_dimension, timestep_dimension),
        #    nn.SiLU(),
        #    nn.Linear(timestep_dimension, timestep_dimension)
        #)

        """Position embedding"""
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        self.pos = nn.Parameter(
            mp.FourierFeatures(th.arange(100), 1, 1000, running_units), 
            requires_grad=True
        )
        
        self.first = nn.Linear(self.predcats, running_units)

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
                embed_type=None,
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
            nn.Linear(running_units, 1),
        )
        #self.final[-1].weight = nn.init.normal_(self.final[-1].weight, 0, 50)
        #self.final[-1].bias = nn.init.constant_(self.final[-1].bias, 1000)
    
    def load_weights(self, ckpt):
        self.load_state_dict(th.load(ckpt, map_location=device))
    
    def load_saved_weights(self, rddir, obj, weights_type='model', load_last=False, retain=False):
        regex = f'*{weights_type}*last*wts*' if load_last else f"*{weights_type}*wts*"
        print(f"<MRCOMMENT> Searching for {weights_type} weights with regular expression {regex}")
        possible_weights_path = glob(os.path.join(rddir, "weights", regex))
        
        # Found something
        if len(possible_weights_path) > 0:
            
            # Found only 1 matching file
            if len(possible_weights_path) == 1:
                weights_path = possible_weights_path[0]
                qualifier = 'only'
            
            # Found multiple matching files
            elif len(possible_weights_path) > 1:
                try:
                    weights_path = [m for m in possible_weights_path if 'high' in m][0]
                    qualifier = '"high"'
                except:
                    weights_path = [m for m in glob(possible_weights_path) if 'last' in m][0]
                    qualifier = '"last"'
            
            print(f"<MRCOMMENT> Loading {qualifier} previous {weights_type} weights: {weights_path}")
            obj.load_state_dict(th.load(weights_path, map_location=device, weights_only=False))

        # Found nothing
        else:
            print(f"Found no weights fitting regular expression")

    def configure_diffusion(self, svdir, amod_dic):
        if type(svdir)==str:
            yaml_file = os.path.join(svdir, "yaml", "config.yaml")
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            diff_config = config['decoder_mdlm']['diffusion_config']
            model = Seq2SeqMDLM(
                encoder_config = config['encoder_dict'],
                decoder_config = config['decoder_diff']['model_config'],
                diff_config = diff_config,
                top_peaks = config['top_peaks'], 
                max_peptide_length = config['pep_length'][1], 
                token_dict = amod_dic,
                ensemble_config   = config['decoder_diff']['ensemble'],
                masses_path = config['loader']['masses_path'],
            )
        else:
            model = svdir
        self.diff_obj = model.diff_obj
        #self.load_saved_weights(svdir, model, 'model', False, False)
        self.predcats = model.decoder.predcats
    
    def get_backprop_prop(self, pred, precursor_mz, precursor_charge):
        pred_mass = pred * self.data_std + self.data_mean
        target_mass = utils.mztomass(precursor_mz, precursor_charge)
        target = (target_mass-self.data_mean) / self.data_std
        loss = (pred - target).square()
        return loss

    def sample_xt_from_x0(self, x0):
        bs, sl = x0.shape
        t = self.diff_obj._sample_t(bs, x0.device)
        sigma, dsigma = self.diff_obj.noise(t)
        model_kwargs = {}
        model_kwargs['timesteps'] = sigma if self.diff_obj.time_conditioning else th.zeros_like(sigma)
        move_chance = 1 - th.exp(-sigma[:, None])
        xt = self.diff_obj.q_xt(x0, move_chance)
        return xt, model_kwargs

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

    def forward(self, latent, **kwargs):
        """
        Model is built differently
        """

        # Time embedding
        time_emb = None #self.time_embed(mp.FourierFeatures(timesteps, 1, 10000, self.timestep_dimension))
        
        # Process latent
        one_hot = nn.functional.one_hot(latent.type(th.int64), self.predcats).type(th.float32)
        one_hot.requires_grad=True
        latent_ = self.first(one_hot) + self.alpha * self.pos[:one_hot.shape[1]]
        out = self.Main(latent_, time_emb)
        
        # Logits
        out = self.final(out)

        return {'out': out.mean(1).squeeze(-1), 'one_hot': one_hot}
