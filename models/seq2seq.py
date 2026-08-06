import torch as th
from torch import nn
from torch.nn import functional as F
from models.encoder import Encoder
from models.foundational_encoder import Encoder as FoundationalEncoder
from models.diff_decoder import DenovoDiffusionDecoder, MDLMDecoder, D3PMDecoder
from models.decoder import DenovoDecoder
from models.diffusion.model_utils import create_diffusion
from models.mdlm.diffusion import Diffusion as MDLMDiffusion
from models.d3pm import D3PM
import os
from copy import deepcopy
import numpy as np

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
total_aa_mass = lambda m_z, charge: (m_z - 1.00727646688)*charge - 18.010565
reinforcement_subs = list("ACDEFGHIKLMNPQRSTVWY")
def find_winners(seqs, masses_ref, exp_mz, charges, n, mass_tol, return_full=False):
    bs = seqs.shape[0] // n
    seqs_rs = seqs.reshape(bs, n, -1)
    ls = [seqs_rs[i].unique(dim=0, return_inverse=True, return_counts=True) for i in range(bs)]
    
    rs = 0
    inds = []
    for m in range(bs):
        inds.append(ls[m][1]+rs)
        rs += int(ls[m][1].max()) + 1
    inds = th.cat(inds, dim=0)
    counts = th.cat([l[2] for l in ls], 0)

    # Does the mass match the precursor?
    masses = masses_ref.to(seqs.device)[None].repeat([bs*n, 1]).gather(1, seqs).sum(-1)
    passfail = abs(masses - total_aa_mass(exp_mz, charges)) < mass_tol

    # Top occurring sequence for each batch member
    cnt_full = counts[inds].reshape(bs, n)
    pf_full = passfail.reshape(bs, n)
    add_index = cnt_full*pf_full
    add_index = add_index.argsort(1) if return_full else add_index.argmax(1)
    
    # Highest occurring sequence when nothing fits precursor
    if return_full==False:
        all_fail = pf_full.sum(1) == 0
        add_index[all_fail] = cnt_full[all_fail].argmax(1)

    # Best index for every batch member
    winners = th.arange(0, bs*n, n).to(seqs.device)
    if return_full:
        winners = (winners[:,None].tile([1,n]) + add_index).reshape(-1,)
    else:
        winners = winners + add_index
        assert len(winners) == bs, f'There should be {bs} winners, not {len(winners)}'
    
    return winners

def calculate_entropy(trajectory_logits):
    batch_size, traj_size, sequence_length, logits_size = trajectory_logits.shape

    traj = trajectory_logits.softmax(dim=-1) # bs, traj, sl, logits
    entropies = (-traj*traj.log()).sum(-1).mean(1)
    #mask = th.arange(sequence_length, device=trajectory.device)[None].tile([batch_size, 1]) <= peptide_length[:,None]
    #peptide_entropies = (entropies*mask).sum(1) / peptide_length

    return entropies

def reshape_top_k(tensor, k):
    shape = tensor.shape
    if len(shape) == 1:
        return tensor.reshape(-1,k)
    elif len(shape) == 2:
        a,b = shape
        return tensor.reshape(-1,k,b)
    elif len(shape) == 3:
        a,b,c = shape
        return tensor.reshape(-1,k,b,c)
    elif len(shape) == 4:
        a,b,c,d = shape
        return tensor.reshape(-1,k,b,c,d)

def expand_batch(batch, n=1):
    bs, sl = batch['mz'].shape
    batch['mz'] = batch['mz'][:,None].tile(1, n, 1).reshape(-1, sl)
    batch['ab'] = batch['ab'][:,None].tile(1, n, 1).reshape(-1, sl)
    batch['charge'] = batch['charge'][:,None].tile(1, n).reshape(-1)
    batch['mass'] = batch['mass'][:,None].tile(1, n).reshape(-1)
    if 'length' in batch:
        batch['length'] = batch['length'][:,None].tile(1, n).reshape(-1)
    if 'peplen' in batch:
        batch['peplen'] = batch['peplen'][:,None].tile(1, n).reshape(-1)
    return batch

def mass_objects(masses_path, output_dictionary):
    path = os.path.join(masses_path, 'masses.tsv')
    str2mass = {
        m.split()[0]: float(m.split()[1]) for m in open(path).read().strip().split("\n")
    }
    int2mass = {Int: str2mass.get(string, 0) for string, Int in output_dictionary.items()}
    masses_array = th.tensor([m[1] for m in sorted(int2mass.items())])
    return str2mass, int2mass, masses_array

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder_config,
        top_peaks,
        encoder_cls=Encoder,
        encoder_model=None,
        **kwargs
    ):
        super(Seq2Seq, self).__init__()
        self.encoder_dict = encoder_config
        self.use_encoder = encoder_config['empty']==False
        if not self.use_encoder:
            print("<S2SCOMMENT> Unconditional decoder - no spectrum encoding")
        
        if encoder_model == None:
            self.encoder = encoder_cls(
                sequence_length=top_peaks,
                device=device,
                **encoder_config,
            ) if self.use_encoder else None
        else:
            self.encoder = encoder_model
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])
    
    def encinp(
        self, 
        batch, 
        mask_length=True, 
        return_mask=False, 
    ):

        mzab = th.cat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab.to(device),
            'charge': (
                batch['charge'] if self.encoder.use_charge else None
            ),
            'mass': (
                batch['mass'] if self.encoder.use_mass else None
            ),
            'length': batch['length'] if mask_length else None,
            'return_mask': return_mask,
        }

        return model_inp       
    
    def encoder_embedding(self, batch):
        encoder_input = self.encinp(batch)
        embedding = self.encoder(**encoder_input)
        return embedding

    def make_reference_model(self, freeze=True):
        self.refmodel = deepcopy(self.decoder)
        if freeze:
            for param in self.refmodel.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        pass

    def predict_sequence(self, *args, **kwargs):
        pass

class Seq2SeqAR(Seq2Seq):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        top_peaks,
        token_dict,
        encoder_model,
        **kwargs,
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
            encoder_cls=FoundationalEncoder,
            encoder_model=encoder_model,
        )
        decoder_config['kv_indim'] = self.encoder.run_units
        self.decoder = DenovoDecoder(
            token_dict=token_dict,
            dec_config=decoder_config,
            encoder=self.encoder,
        )

    def forward(self,
        intseq,
        batch,
        causal=False,
        training=False,
        softmax=False,
    ):
        embedding = self.encoder_embedding(batch)
        logits = self.decoder(intseq, embedding, batch)
        return logits

    def predict_sequence(self, batch, **kwargs):
        embedding = self.encoder_embedding(batch)
        out_dict = self.decoder.predict_sequence(embedding, batch)
        return out_dict

class Seq2SeqDiff(Seq2Seq):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        diff_config,
        ensemble_config,
        top_peaks,
        token_dict,
        **kwargs
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
        )
        self.diff_config = diff_config
        decoder_config['kv_indim'] = self.encoder.run_units
        self.diff_obj = create_diffusion(**diff_config)
        self.decoder = DenovoDiffusionDecoder(
            input_output_units = diff_config['in_channel'], # perhaps replace this with running units
            clip_denoised      = diff_config['clip_denoised'],
            output_sigma       = diff_config['learn_sigma'],
            token_dict         = token_dict,
            dec_config         = decoder_config,
            diff_obj           = self.diff_obj,
            **decoder_config,
        )

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        # Scale
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)

    def condition_function(self, classifier, latent, t, class_index, scale):
        latent.requires_grad = True
        out = classifier(latent, t)[:, class_index]
        out.mean().backward()
        return latent.grad * scale

    def forward_eval(self, batch, save_xcur=False, save_xstart=False, cond_fn=None, progress=False):
        embedding = self.encoder_embedding(batch)
        output = self.decoder.predict_sequence(
            embedding, 
            batch, 
            save_xcur=save_xcur, 
            save_xstart=save_xstart, 
            cond_fn=cond_fn,
            progress=progress,
        )
        return output

    def forward(self, batch, target, global_step, timesteps):
        embedding = self.encoder_embedding(batch)
        model_kwargs = {
            'input_ids': None,
            'decoder_input_ids': target,
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_feats': embedding['emb'],
        }
        if self.diff_config['use_loss_mask']:
            model_kwargs['loss_mask'] = loss_mask # THIS RUINS EVERYTHING
        losses = self.diff_obj.training_losses(
            self.decoder,
            global_step,
            timesteps,
            model_kwargs=model_kwargs,
            noise=None
        )
        return losses

    def predict_sequence(
        self,
        batch,
        save_xcur=False,
        save_xstart=True,
        entropy=True, # replace logits with entropy calculation
        n=None,
        return_full=False,
        cls_dict=None,
        progress=False,
    ):
        bs, sl = batch['mz'].shape
        n = self.ens_size if n==None else n
        cond_fn = (
            None if cls_dict == None else
            lambda latent, t: self.condition_function(cls_dict['model'], latent, t, cls_dict['index'], cls_dict['scale']) 
        )

        full_size = bs*n
        batch = expand_batch(batch, n=n)
        diffout = self.forward_eval(
            batch, 
            save_xcur=save_xcur, 
            save_xstart=save_xstart, 
            cond_fn=cond_fn, 
            progress=progress,
        )
        # Depending on arguments, the output of the decoder will differ
        seqs = diffout.pop('prediction')
        logits = diffout.pop('logits')
        if entropy:
            trajectory_logits = self.decoder.get_logits(diffout['xstart'].detach())
            diffout['entropy'] = calculate_entropy(trajectory_logits)

        winners = find_winners(
            seqs, self.masses, batch['mass'], batch['charge'], n, self.mass_tol, return_full=return_full
        )

        reshape = (lambda x: reshape_top_k(x, n)) if return_full else lambda x: x
        top_sequences = reshape(seqs[winners])
        logits = reshape(logits[winners])
        additional_outputs = {x: reshape(y[winners]) for x,y in diffout.items()}

        return_ = {'prediction': top_sequences, 'logits': logits} | additional_outputs
        return return_       

class Seq2SeqMDLM(Seq2Seq):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        diff_config,
        ensemble_config=None,
        top_peaks=100,
        token_dict={},
        **kwargs
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
        )
        # Decoder model
        decoder_config['kv_indim'] = self.encoder.run_units if self.use_encoder else None
        decoder_config['embed_type'] = 'preembed' if diff_config['time_conditioning'] else None
        self.decoder = MDLMDecoder(
            token_dict          = token_dict,
            decoder_config      = decoder_config,
            use_guidance        = diff_config['guidance']['p_uncond']>0,
            **decoder_config,
        )
        # Diffusion object
        self.diff_config = diff_config
        self.diff_obj = MDLMDiffusion(diff_config, self.decoder.outdict, self.decoder)
        self.decoder.diff_obj = self.diff_obj

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        # Scale
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)
            self.masses = self.masses.to(device)
            #self.masses = nn.Parameter(self.masses, requires_grad=False)

        self.rl_subs = [value for key, value in token_dict.items() if key[0] in reinforcement_subs]
    
    def get_reveal_steps(self, x_in_time):
        trajectory_length = x_in_time.shape[1]
        reveal = ((x_in_time != self.decoder.MASK).int().argmax(1)-1).clip(min=0)
        #never_selected = x_in_time[:, -1] == self.decoder.MASK
        #reveal[never_selected] = trajectory_length - 2
        return reveal

    def calculate_min_peptide_prob(self, prediction, logits_in_time, sl_mask):
        bs, steps, sl, cats = logits_in_time.shape
        min_conf_ = logits_in_time.gather(-1, prediction[:,None,:,None].tile([1,steps,1,1]))[...,0].min(dim=1)[0]
        return min_conf_, (min_conf_*sl_mask).sum(dim=-1) / (sl_mask.sum(dim=-1)+1e-9)

    def calculate_entropy_prob(self, logits_in_time, reveal_mask, sl_mask):
        entropy = -(logits_in_time * (logits_in_time+1e-9).log()).sum(dim=-1)
        aa_entropy = (entropy*reveal_mask).sum(dim=1) / (reveal_mask.sum(dim=1)+1e-9) # average over masked tokens
        pep_entropy = (aa_entropy*sl_mask).sum(dim=-1) / (sl_mask.sum(dim=-1)+1e-9) # average over sequence length
        return aa_entropy, pep_entropy
    
    def dropout_attributes(self, batch):
        batch_size = batch['mass'].shape[0]
        if self.diff_obj.use_guidance:
            dropout_mask = th.rand(batch_size) < self.diff_config['guidance']['p_uncond']
            if 'mass' in batch: batch['mass'][dropout_mask] = 0.
            if 'charge' in batch: batch['charge'][dropout_mask] = 0
            #if 'kv_features' in batch: batch['kv_features'][dropout_mask] = 0.
        return batch

    def forward_eval(self, batch, top=None, save_x=False, save_p=False, num_steps=None, progress=False, **kwargs):
        if self.use_encoder:
            dictionary = self.encoder_embedding(batch)
            embedding = dictionary['emb']
            spectrum_mask = dictionary['mask']
        else:
            embedding = spectrum_mask = None
        decout = self.decoder.predict_sequence(embedding, batch, top=top, save_x=save_x, save_p=save_p, num_steps=num_steps, progress=progress, **kwargs)
        return decout

    def forward(self, batch, target, training_mask=None, block_decoding=False, rl=False):
        forward_function = self.contrastive_loss if rl else self.decoder.diff_obj._forward_pass_diffusion
        if self.use_encoder:
            dictionary = self.encoder_embedding(batch)
            embedding = dictionary['emb']
            spectrum_mask = dictionary['mask']
        else:
            embedding = spectrum_mask = None
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding,
            'seqmask': training_mask,
            'doubled': True if block_decoding else False,
        }
        model_kwargs = self.dropout_attributes(model_kwargs)
        return forward_function(target, model_kwargs, block_decoding)

    def predict_sequence(
        self, 
        batch: dict,                # batch of inputs
        batch_size: int=None,       # batch_size; to be used for unconditional mode
        save_x: bool=False,         # return the intseqs at every step
        save_p: bool=False,         # return the logits at every step
        num_steps: int=None,        # number of sampling steps in decoder
        top: int=None,              # top categorical sampling; None defaults to config.yaml setting
        n: int=None,                # return n sequences per batch member; None defaults to config.yaml setting
        return_full: dict=False,    # return n outputs for each batch member (instead of 1/top sequence)
        progress: bool=False,       # tqdm progress bar
        guide_model: nn.Module=None,# guidance_model
        gamma: int=1.,              # scaler for guidance
    ):
        # Input batch
        batch_size, SL = batch['mz'].shape if 'mz' in batch else (batch['charge'].shape[0] if 'charge' in batch else batch_size)
        n = self.ens_size if n==None else n
        batch = expand_batch(batch, n=n)
        
        # Model outputs
        diffout = self.forward_eval(
            batch,
            top=top,
            save_x=save_x,
            save_p=save_p,
            num_steps=num_steps,
            progress=progress,
            guide_model=guide_model,
            gamma=gamma,
            batch_size=batch_size,
        )
        seqs = diffout.pop('prediction')
        logits = diffout.pop('logits')
        
        # Probability calculations
        if save_p and save_x:
            nbs, sl = seqs.shape
            slmask = th.arange(sl, device=device)[None].tile([nbs, 1]) < (seqs == self.decoder.EOS).int().argmax(dim=1)[:,None]
            diffout['aa_prob_min'], diffout['pep_prob_min'] = self.calculate_min_peptide_prob(seqs, diffout['p_save'], slmask)
            
            reveal = self.get_reveal_steps(diffout['x_save'])
            reveal_mask = th.arange(diffout['p_save'].shape[1], device=device)[None,:,None].tile([nbs, 1, sl]) < reveal[:,None]
            diffout['aa_entropy'], diffout['pep_entropy'] = self.calculate_entropy_prob(diffout['p_save'], reveal_mask, slmask)
        
        # Find winners
        if n == 1:
            winners = th.arange(batch_size)
        else:
            winners = find_winners(
                seqs, self.masses, batch['mass'], batch['charge'], n, self.mass_tol, return_full=return_full
            )
        
        # Select winners and reshape
        reshape = (lambda x: reshape_top_k(x, n)) if return_full else lambda x: x
        top_sequences = reshape(seqs[winners])
        logits = reshape(logits[winners])
        additional_outputs = {x: reshape(y[winners]) for x, y in diffout.items()}

        return_ = {'prediction': top_sequences, 'logits': logits} | additional_outputs
        return return_

    def generate_negative(self, xt, x0, model_kwargs, n=1):
        bs, sl = xt.shape
        _, sl2, kvs = model_kwargs['kv_features'].shape
        if 'self_conditions' in model_kwargs:
            _, _, cats = model_kwargs['self_conditions'].shape
        
        xt = xt[:,None].tile([1, n, 1]).reshape(n*bs, sl)
        embedding = model_kwargs['kv_features'][:,None].tile([1, n, 1, 1]).reshape(n*bs, sl2, kvs)
        batch = {
            'mass': model_kwargs['mass'][:,None].tile([1, n]).reshape(n*bs),
            'charge': model_kwargs['charge'][:,None].tile([1, n]).reshape(n*bs),
            #'kv_features': model_kwargs['kv_features'][:,None].tile([1, n, 1, 1]).reshape(n*bs, sl2, kvs)
            'self_conditions': model_kwargs['self_conditions'][:,None].tile([1, n, 1, 1]).reshape(n*bs, sl, cats),
        }
        with th.no_grad():
            decout = self.decoder.predict_sequence(embedding, batch, x_init=xt, top=100,)
            prediction = decout['prediction']

        original_masses = self.masses.to(x0.device)[None].tile([x0.shape[0], 1]).gather(-1, x0).sum(-1)
        masses = self.masses.to(x0.device)[None].tile([prediction.shape[0], 1]).gather(-1, prediction).sum(-1)
        deltas = (original_masses[:,None] - masses.reshape(bs, n)).abs()
        losers = ((deltas!=0).float() + th.rand_like(deltas)).argmax(-1)
        xl = prediction.reshape(bs, n, -1)[th.arange(bs), losers]
        delta_masses = deltas[th.arange(bs), losers]
        anys = (deltas!=0).any(1)
        xl = th.where(anys[:,None], xl, x0)
        
        # Second bests for same solutions
        bsinds = th.where(~anys)[0]
        peptide_lengths = th.where(x0==self.decoder.EOS)[1]
        longenough = th.arange(sl, device=xt.device)[None].tile([bs, 1]) < peptide_lengths[:,None]
        slinds = (((xt.reshape(bs,n,-1)[:,0]==self.decoder.MASK)&(longenough)).float() + th.rand(bs, sl,device=xt.device)).argmax(1)[~anys]
        #xl[bsinds, slinds] = decout['logits'].reshape(bs, n, sl, -1)[bsinds, 0, slinds]
        subs = decout['logits'].reshape(bs, n, sl, -1)[bsinds, 0, slinds].argsort(1)[:,-2]
        xl[bsinds,slinds] = subs
        delta_masses = (original_masses - self.masses.to(x0.device)[None].tile([xl.shape[0], 1]).gather(-1, xl).sum(-1)).abs()
        
        return xl, delta_masses

    def contrastive_loss(self, x0, model_kwargs, *args, beta=0.1, **kwargs):

        # Set up all variables: xt, timesteps, self_conditions
        bs, sl = x0.shape
        device = x0.device
        t = self.diff_obj._sample_t(bs, device).clamp(0.2)
        sigma, dsigma = self.diff_obj.noise(t)
        model_kwargs['timesteps'] = sigma if self.diff_obj.time_conditioning else th.zeros_like(sigma)
        move_chance = 1 - th.exp(-sigma[:, None])
        xt = self.diff_obj.q_xt(x0, move_chance) # th.full_like(x0, self.decoder.MASK)
        #xl, xt, delta_mass = self.generate_negative(xt, x0, sub_rate=.2)
        
        if self.diff_obj.config['model']['self_condition']:
            model_kwargs['self_conditions'] = th.zeros(xt.shape[0], xt.shape[1], self.diff_obj.vocab_size, device=device)
            if np.random.uniform() > 0.5:
                with th.no_grad():
                    model_output = self.decoder(xt, **model_kwargs)['out']
                model_kwargs['self_conditions'] = model_output.detach()
        
        xl, delta_mass = self.generate_negative(xt, x0=x0, model_kwargs=model_kwargs, n=4)
        # Get log(probabilities) from both models
        logprobs_policy_ = self.decoder(xt, **model_kwargs)['out']
        logprobs_policy = logprobs_policy_.log_softmax(dim=-1)
        with th.no_grad():
            logprobs_ref = self.refmodel(xt, **model_kwargs)['out'].log_softmax(dim=-1)

        # Extract Log-Probs for the correct categories of masked tokens
        lp_win_policy = logprobs_policy.gather(-1, x0[...,None]).squeeze(-1)
        lp_win_ref = logprobs_ref.gather(-1, x0[...,None]).squeeze(-1)

        # xl
        lp_loss_policy = logprobs_policy.gather(-1, xl[...,None]).squeeze(-1)
        lp_loss_ref = logprobs_ref.gather(-1, xl[...,None]).squeeze(-1)

        # DPO Contrastive Objective
        first_term = lp_win_policy - lp_loss_policy
        second_term = lp_win_ref - lp_loss_ref
        first_term_ = lp_win_policy - lp_win_ref
        second_term_ = lp_loss_policy - lp_loss_ref
        logits = first_term - second_term
        # KL Divergence
        kl = (logprobs_policy.detach().exp() * (logprobs_policy.detach() - logprobs_ref)).sum(-1).mean().item()
        
        weights = th.log1p(delta_mass.abs())
        masked_mask = (xt == self.decoder.MASK) & (x0!=xl)
        masked_mask_ = (xt == self.decoder.MASK) & (x0==xl)
        cross_entropy = F.cross_entropy(logprobs_policy.transpose(-1,-2), x0, reduction='none')[masked_mask_].mean()
        loss_ = -F.logsigmoid(beta * logits) # (win-loss) must be larger for lower beta to get same loss as larger beta
        loss_ = (weights[:,None] * loss_)[masked_mask].mean()
        loss = loss_ + 0*cross_entropy

        wvl = first_term[masked_mask].detach().mean().item()
        wvl_ref = second_term[masked_mask].detach().mean().item()
        relative_win = first_term_[masked_mask].detach().mean().item()
        relative_loss = second_term_[masked_mask].detach().mean().item()
        
        return {
            'loss': loss,
            'logsigmoid': loss_.item(),
            'cross_entropy': cross_entropy.item(),
            'win_vs_loss': wvl,
            'win_vs_loss_ref': wvl_ref,
            'relative_win': relative_win,
            'relative_loss': relative_loss,
            'kl': kl,
        }

class Seq2SeqD3PM(Seq2Seq):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        diff_config,
        ensemble_config=None,
        top_peaks=100,
        token_dict={},
        **kwargs
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
        )
        # Decoder model
        decoder_config['kv_indim'] = self.encoder.run_units
        decoder_config['wavelength_bounds'] = (1, 5*diff_config['steps'])
        decoder_config['embed_type'] = 'preembed'
        self.decoder = D3PMDecoder(
            token_dict = token_dict,
            decoder_config = decoder_config,
            **decoder_config,
        )
        # Diffusion object
        self.diff_obj = D3PM(
            x0_model=self.decoder,
            n_T=diff_config['steps'],
            num_classes=(self.decoder.predcats),
        )
        self.decoder.diff_obj = self.diff_obj

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        # Scale
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)
    
    def calculate_min_peptide_prob(self, prediction, logits_in_time, sl_mask):
        bs, steps, sl, cats = logits_in_time.shape
        min_conf_ = logits_in_time.gather(-1, prediction[:,None,:,None].tile([1,steps,1,1]))[...,0].min(dim=1)[0]
        return min_conf_, (min_conf_*sl_mask).sum(dim=-1) / (sl_mask.sum(dim=-1)+1e-9)

    def calculate_entropy_prob(self, logits_in_time, reveal_mask, sl_mask):
        entropy = -(logits_in_time * (logits_in_time+1e-9).log()).sum(dim=-1)
        aa_entropy = (entropy*reveal_mask).sum(dim=1) / (reveal_mask.sum(dim=1)+1e-9) # average over masked tokens
        pep_entropy = (aa_entropy*sl_mask).sum(dim=-1) / (sl_mask.sum(dim=-1)+1e-9) # average over sequence length
        return aa_entropy, pep_entropy

    def forward(self, batch, top=None, save_x=False, save_p=False, num_steps=None, progress=False, **kwargs):
        dictionary = self.encoder_embedding(batch)
        embedding = dictionary['emb']
        spectrum_mask = dictionary['mask']
        decout = self.decoder.predict_sequence(embedding, batch, top=top, save_x=save_x, save_p=save_p, num_steps=num_steps, progress=progress)
        return decout
    
    def predict_sequence(
        self, 
        batch: dict,             # batch of inputs
        save_x: bool=False,      # return the intseqs at every step
        save_p: bool=False,      # return the logits at every step
        num_steps: int=None,     # number of sampling steps in decoder
        top: int=None,           # top categorical sampling; None defaults to config.yaml setting
        n: int=None,             # return n sequences per batch member; None defaults to config.yaml setting
        return_full: dict=False, # return n outputs for each batch member (instead of 1/top sequence)
        progress: bool=False,    # tqdm progress bar
    ):
        # Input batch
        batch_size, SL = batch['mz'].shape
        n = self.ens_size if n==None else n
        batch = expand_batch(batch, n=n)
        
        # Model outputs
        diffout = self(batch, top=top, save_x=save_x, save_p=save_p, num_steps=num_steps, progress=progress)
        seqs = diffout.pop('prediction')
        logits = diffout.pop('logits')
        
        # Probability calculations
        if save_p and save_x:
            nbs, sl = seqs.shape
            slmask = th.arange(sl, device=device)[None].tile([nbs, 1]) < (seqs == self.decoder.EOS).int().argmax(dim=1)[:,None]
            diffout['aa_prob_min'], diffout['pep_prob_min'] = self.calculate_min_peptide_prob(seqs, diffout['p_save'], slmask)
            
            #reveal = self.get_reveal_steps(diffout['x_save'])
            #reveal_mask = th.arange(diffout['p_save'].shape[1], device=device)[None,:,None].tile([nbs, 1, sl]) < reveal[:,None]
            #diffout['aa_entropy'], diffout['pep_entropy'] = self.calculate_entropy_prob(diffout['p_save'], reveal_mask, slmask)
        
        # Find winners
        if n == 1:
            winners = th.arange(batch_size)
        else:
            winners = find_winners(
                seqs, self.masses, batch['mass'], batch['charge'], n, self.mass_tol, return_full=return_full
            )
        
        # Select winners and reshape
        reshape = (lambda x: reshape_top_k(x, n)) if return_full else lambda x: x
        top_sequences = reshape(seqs[winners])
        logits = reshape(logits[winners])
        additional_outputs = {x: reshape(y[winners]) for x, y in diffout.items()}

        return_ = {'prediction': top_sequences, 'logits': logits} | additional_outputs
        return return_
