import torch as th
from torch import nn
from copy import deepcopy
from models.encoder import Encoder
from models.diff_decoder import DenovoDiffusionDecoder, MDLMDecoder
from models.decoder import DenovoDecoder
from models.diffusion.model_utils import create_diffusion
from models.mdlm.diffusion import Diffusion as MDLMDiffusion
from models.d3pm.diffusion import D3PMDiffusion
from models.insertdelete.diffusion import InsertDeleteDiffusion
import os

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
total_aa_mass = lambda m_z, charge: (m_z - 1.00727646688)*charge - 18.010565
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


def _resolve_time_conditioning_and_embed_type(diff_config, decoder_tag):
    time_conditioning = bool(diff_config.get('time_conditioning', True))
    embed_type = diff_config.get('model', {}).get('embed_type', None)
    if time_conditioning and embed_type is None:
        raise ValueError(
            f"{decoder_tag}: diffusion_config.time_conditioning=True requires "
            "diffusion_config.model.embed_type to be set in yaml "
            "(e.g. 'adaLN', 'normembed', or 'preembed')."
        )
    if not time_conditioning:
        return False, None
    return True, embed_type


def _ensure_insertdelete_tokens(token_dict):
    out = deepcopy(token_dict)
    next_idx = max(out.values()) + 1
    for token_name in ["<INS>", "<DEL>"]:
        if token_name not in out:
            out[token_name] = next_idx
            next_idx += 1
    return out

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder_config,
        top_peaks,
        **kwargs
    ):
        super(Seq2Seq, self).__init__()
        self.encoder_dict = encoder_config
        self.use_precomputed_encoder = kwargs.get('use_precomputed_encoder', False)
        self.precomputed_encoder_dim = kwargs.get('precomputed_encoder_dim', None)
        self.precomputed_kv_indim = kwargs.get('precomputed_kv_indim', None)
        self.encoder = None
        if not self.use_precomputed_encoder:
            self.encoder = Encoder(
                sequence_length=top_peaks,
                device=device,
                **encoder_config,
            )
        self.precomputed_proj = None

    def configure_precomputed_encoder(self, target_kv_indim):
        self.precomputed_kv_indim = int(target_kv_indim)
        if not self.use_precomputed_encoder:
            self.precomputed_proj = None
            return
        if self.precomputed_encoder_dim is None:
            self.precomputed_proj = None
            return
        if int(self.precomputed_encoder_dim) != int(self.precomputed_kv_indim):
            self.precomputed_proj = nn.Linear(
                int(self.precomputed_encoder_dim),
                int(self.precomputed_kv_indim),
                bias=False,
            )
        else:
            self.precomputed_proj = None
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])
    
    def encinp(
        self, 
        batch, 
        mask_length=True, 
        return_mask=False, 
    ):
        if self.encoder is None:
            raise RuntimeError("encinp() called without an initialized encoder.")

        mzab = th.cat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab,
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
        if self.use_precomputed_encoder:
            if 'enc_emb' not in batch:
                raise KeyError("Expected 'enc_emb' in batch when use_precomputed_encoder=True")
            emb = batch['enc_emb']
            if self.precomputed_proj is not None:
                emb = self.precomputed_proj(emb)
            elif emb.shape[-1] != self.precomputed_kv_indim:
                raise RuntimeError(
                    f"enc_emb dim {emb.shape[-1]} does not match precomputed_kv_indim {self.precomputed_kv_indim}"
                )
            mask = batch.get('enc_mask', None)
            if mask is not None:
                if mask.dtype == th.bool:
                    mask = mask.type(th.float32) * 1e7
                else:
                    mask = mask.type(th.float32)
                    if float(mask.max().detach().cpu().item()) <= 1.0:
                        mask = mask * 1e7
            return {'emb': emb, 'mask': mask, 'other': None}
        encoder_input = self.encinp(batch)
        embedding = self.encoder(**encoder_input)
        return embedding

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
        **kwargs,
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
            **kwargs,
        )
        if self.use_precomputed_encoder:
            raise NotImplementedError("Seq2SeqAR does not support use_precomputed_encoder=True.")
        decoder_kv_indim = decoder_config.get('kv_indim', self.encoder.run_units)
        decoder_config['kv_indim'] = decoder_kv_indim
        self.configure_precomputed_encoder(decoder_kv_indim)
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

    def predict_sequence(self, batch):
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
            **kwargs,
        )
        default_kv_indim = encoder_config.get('running_units')
        if self.encoder is not None:
            default_kv_indim = self.encoder.run_units
        decoder_kv_indim = decoder_config.get('kv_indim', default_kv_indim)
        decoder_config['kv_indim'] = decoder_kv_indim
        self.configure_precomputed_encoder(decoder_kv_indim)
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

    def forward(self, batch, save_xcur=False, save_xstart=False, cond_fn=None, progress=False):
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
        diffout = self(
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
            **kwargs,
        )
        # Decoder model
        default_kv_indim = encoder_config.get('running_units')
        if self.encoder is not None:
            default_kv_indim = self.encoder.run_units
        decoder_kv_indim = decoder_config.get('kv_indim', default_kv_indim)
        decoder_config['kv_indim'] = decoder_kv_indim
        self.configure_precomputed_encoder(decoder_kv_indim)
        _, embed_type = _resolve_time_conditioning_and_embed_type(diff_config, "MDLM")
        self.decoder = MDLMDecoder(
            token_dict          = token_dict,
            decoder_config      = decoder_config,
            embed_type          = embed_type,
            **decoder_config,
        )
        # Diffusion object
        self.diff_obj = MDLMDiffusion(diff_config, self.decoder.outdict, self.decoder)
        self.decoder.diff_obj = self.diff_obj

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        # Scale
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)
    
    def get_reveal_steps(self, x_in_time):
        trajectory_length = x_in_time.shape[1]
        reveal = ((x_in_time != self.decoder.MASK).int().argmax(1)-1).clip(min=0)
        never_selected = x_in_time[:, -1] == self.decoder.MASK
        reveal[never_selected] = trajectory_length - 2
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
            slmask = th.arange(sl, device=seqs.device)[None].tile([nbs, 1]) < (seqs == self.decoder.EOS).int().argmax(dim=1)[:,None]
            diffout['aa_prob_min'], diffout['pep_prob_min'] = self.calculate_min_peptide_prob(seqs, diffout['p_save'], slmask)
            
            reveal = self.get_reveal_steps(diffout['x_save'])
            reveal_mask = th.arange(diffout['p_save'].shape[1], device=seqs.device)[None,:,None].tile([nbs, 1, sl]) < reveal[:,None]
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
            **kwargs,
        )
        # Decoder model
        default_kv_indim = encoder_config.get('running_units')
        if self.encoder is not None:
            default_kv_indim = self.encoder.run_units
        decoder_kv_indim = decoder_config.get('kv_indim', default_kv_indim)
        decoder_config['kv_indim'] = decoder_kv_indim
        self.configure_precomputed_encoder(decoder_kv_indim)
        _, embed_type = _resolve_time_conditioning_and_embed_type(diff_config, "D3PM")
        self.decoder = MDLMDecoder(
            token_dict          = token_dict,
            decoder_config      = decoder_config,
            embed_type          = embed_type,
            **decoder_config,
        )
        # Diffusion object
        self.diff_obj = D3PMDiffusion(diff_config, self.decoder.outdict, self.decoder)
        self.decoder.diff_obj = self.diff_obj

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        # Scale
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)
    
    def get_reveal_steps(self, x_in_time):
        trajectory_length = x_in_time.shape[1]
        reveal = ((x_in_time != self.decoder.MASK).int().argmax(1)-1).clip(min=0)
        never_selected = x_in_time[:, -1] == self.decoder.MASK
        reveal[never_selected] = trajectory_length - 2
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
            slmask = th.arange(sl, device=seqs.device)[None].tile([nbs, 1]) < (seqs == self.decoder.EOS).int().argmax(dim=1)[:,None]
            diffout['aa_prob_min'], diffout['pep_prob_min'] = self.calculate_min_peptide_prob(seqs, diffout['p_save'], slmask)
            
            reveal = self.get_reveal_steps(diffout['x_save'])
            reveal_mask = th.arange(diffout['p_save'].shape[1], device=seqs.device)[None,:,None].tile([nbs, 1, sl]) < reveal[:,None]
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


class Seq2SeqInsertDelete(Seq2Seq):
    """Seq2Seq wrapper for insert/delete diffusion decoding.

    This class wires the shared encoder + MDLM decoder backbone with the extra
    insert/delete heads and the `InsertDeleteDiffusion` objective/sampler.
    """

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
        """Construct insert/delete seq2seq model.

        Args:
            encoder_config: Encoder configuration dict.
            decoder_config: Decoder configuration dict.
            diff_config: Insert/delete diffusion configuration.
            ensemble_config: Inference ensembling settings.
            top_peaks: Number of spectrum peaks for encoder input.
            token_dict: Base output token dictionary.
            **kwargs: Optional runtime/data settings (masses path, precomputed encoder args).
        """
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
            **kwargs,
        )
        token_dict = _ensure_insertdelete_tokens(token_dict)

        default_kv_indim = encoder_config.get('running_units')
        if self.encoder is not None:
            default_kv_indim = self.encoder.run_units
        decoder_kv_indim = decoder_config.get('kv_indim', default_kv_indim)
        decoder_config['kv_indim'] = decoder_kv_indim
        self.configure_precomputed_encoder(decoder_kv_indim)
        _, embed_type = _resolve_time_conditioning_and_embed_type(diff_config, "InsertDelete")
        self.decoder = MDLMDecoder(
            token_dict          = token_dict,
            decoder_config      = decoder_config,
            embed_type          = embed_type,
            **decoder_config,
        )

        # Extra heads required by the insert/delete reverse-process marginals.
        run_units = int(decoder_config['running_units'])
        max_len = int(decoder_config['sequence_length'])
        self.decoder.insertdelete_insert_head = nn.Linear(run_units, 1)
        # Delete-count logits are predicted for token positions plus an explicit
        # EOS boundary position (constructed in InsertDeleteDiffusion).
        self.decoder.insertdelete_delete_head = nn.Linear(run_units, max_len + 1)
        self.decoder.insertdelete_length_head = nn.Linear(run_units, max_len + 1)

        self.diff_obj = InsertDeleteDiffusion(diff_config, self.decoder.outdict, self.decoder)
        self.decoder.diff_obj = self.diff_obj

        self.ens_size = ensemble_config['ensemble_n']
        self.mass_tol = eval(ensemble_config['mass_tol'])
        if 'masses_path' in kwargs:
            self.str2mass, self.int2mass, self.masses = mass_objects(kwargs['masses_path'], self.decoder.outdict)

    def forward(self, batch, top=None, save_x=False, save_p=False, num_steps=None, progress=False, **kwargs):
        """Run one insert/delete diffusion decode pass.

        Args:
            batch: Input batch with spectra/conditioning fields.
            top: Optional top-k sampling parameter.
            save_x: Whether to save intermediate token states.
            save_p: Whether to save intermediate token probabilities.
            num_steps: Optional number of reverse steps.
            progress: Whether to show sampler progress.
            **kwargs: Reserved compatibility kwargs.

        Returns:
            Decoder output dict containing prediction/logits and optional traces.
        """
        dictionary = self.encoder_embedding(batch)
        embedding = dictionary['emb']
        decout = self.decoder.predict_sequence(embedding, batch, top=top, save_x=save_x, save_p=save_p, num_steps=num_steps, progress=progress)
        return decout

    def predict_sequence(
        self,
        batch: dict,
        save_x: bool=False,
        save_p: bool=False,
        num_steps: int=None,
        top: int=None,
        n: int=None,
        return_full: dict=False,
        progress: bool=False,
    ):
        """Predict peptide sequences with optional ensembling and diagnostics.

        Args:
            batch: Input batch dict.
            save_x: Whether to return sampled token trajectories.
            save_p: Whether to return probability trajectories.
            num_steps: Optional number of reverse diffusion steps.
            top: Optional top-k sampling parameter.
            n: Number of ensemble samples per input.
            return_full: If True, keep all ensemble candidates.
            progress: Whether to show sampler progress.

        Returns:
            Dict with `prediction`, `logits`, and optional trajectory diagnostics.
        """
        batch_size, SL = batch['mz'].shape
        n = self.ens_size if n==None else n
        batch = expand_batch(batch, n=n)

        diffout = self(batch, top=top, save_x=save_x, save_p=save_p, num_steps=num_steps, progress=progress)
        seqs = diffout.pop('prediction')
        logits = diffout.pop('logits')

        if n == 1:
            winners = th.arange(batch_size)
        else:
            winners = find_winners(
                seqs, self.masses, batch['mass'], batch['charge'], n, self.mass_tol, return_full=return_full
            )

        reshape = (lambda x: reshape_top_k(x, n)) if return_full else lambda x: x
        top_sequences = reshape(seqs[winners])
        logits = reshape(logits[winners])
        additional_outputs = {x: reshape(y[winners]) for x, y in diffout.items()}

        return_ = {'prediction': top_sequences, 'logits': logits} | additional_outputs
        return return_
