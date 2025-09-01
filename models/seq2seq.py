import torch as th
from torch import nn
from models.encoder import Encoder
from models.diff_decoder import DenovoDiffusionDecoder, MDLMDecoder
from models.decoder import DenovoDecoder
from models.diffusion.model_utils import create_diffusion
from models.mdlm.diffusion import Diffusion as MDLMDiffusion
import os

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
total_aa_mass = lambda m_z, charge: (m_z - 1.00727646688)*charge - 18.010565

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder_config,
        top_peaks,
        **kwargs
    ):
        super(Seq2Seq, self).__init__()
        self.encoder_dict = encoder_config

        self.encoder = Encoder(
            sequence_length=top_peaks,
            device=device,
            **encoder_config,
        )
    
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
        )
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
            path = os.path.join(kwargs['masses_path'], 'masses.tsv')
            self.str2mass = {
                m.split()[0]: float(m.split()[1]) 
                for m in open(path)
                .read().strip().split("\n")
            }
            self.int2mass = {Int: self.str2mass.get(string, 0) for string, Int in self.decoder.outdict.items()}
            self.masses = th.tensor([m[1] for m in sorted(self.int2mass.items())])

    def condition_function(self, classifier, latent, t, class_index, scale):
        latent.requires_grad = True
        out = classifier(latent, t)[:, class_index]
        out.mean().backward()
        return latent.grad * scale
    
    def expand_batch(self, batch, n=None):
        n = self.ens_size if n is None else n
        bs, sl = batch['mz'].shape
        batch['mz'] = batch['mz'][:,None].tile(1, n, 1).reshape(-1, sl)
        batch['ab'] = batch['ab'][:,None].tile(1, n, 1).reshape(-1, sl)
        batch['length'] = batch['length'][:,None].tile(1, n).reshape(-1)
        batch['charge'] = batch['charge'][:,None].tile(1, n).reshape(-1)
        batch['mass'] = batch['mass'][:,None].tile(1, n).reshape(-1)
        batch['peplen'] = batch['peplen'][:,None].tile(1, n).reshape(-1)
        return batch

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

    def calculate_entropy(self, trajectory):
        batch_size, traj_size, sequence_length, logits_size = trajectory.shape

        traj = self.decoder.get_logits(trajectory.detach()).softmax(dim=-1) # bs, traj, sl, logits
        entropies = (-traj*traj.log()).sum(-1).mean(1)
        #mask = th.arange(sequence_length, device=trajectory.device)[None].tile([batch_size, 1]) <= peptide_length[:,None]
        #peptide_entropies = (entropies*mask).sum(1) / peptide_length

        return entropies

    def predict_sequence(
        self,
        batch,
        save_xcur=False,
        save_xstart=True,
        entropy=True, # replace logits with entropy calculation
        n=None,
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
        batch = self.expand_batch(batch, n=n)
        diffout = self(
            batch, 
            save_xcur=save_xcur, 
            save_xstart=save_xstart, 
            cond_fn=cond_fn, 
            progress=progress,
        )
        # Depending on arguments, the output of the decoder will differ
        """if len(diffout) == 4:
            seqs, logits, xcur, xstart = diffout
            additional_outputs = (xcur, xstart)
        elif len(diffout) == 3:
            seqs, logits, xcurstart = diffout

            if entropy:
                logits = self.calculate_entropy(xcurstart)
                additional_outputs = ()
            else:
                additional_outputs = (xcurstart,)
        else:
            seqs, logits = diffout
            additional_outputs = ()"""
        seqs = diffout.pop('prediction')
        logits = diffout.pop('logits')
        if entropy:
            diffout['entropy'] = self.calculate_entropy(diffout['xstart'])
        
        seqs_rs = seqs.reshape(bs, n, -1)
        ls = [seqs_rs[i].unique(dim=0, return_inverse=True, return_counts=True) for i in range(bs)]
        #uniqs = th.cat([l[0] for l in ls], 0)
        rs = 0
        inds = []
        for m in range(bs):
            inds.append(ls[m][1]+rs)
            rs += int(ls[m][1].max()) + 1
        inds = th.cat(inds, dim=0)
        counts = th.cat([l[2] for l in ls], 0)
        
        # Does the mass match the precursor?
        masses = self.masses.to(device)[None].repeat([full_size, 1]).gather(1, seqs).sum(-1)
        passfail = abs(masses - total_aa_mass(batch['mass'], batch['charge'])) < self.mass_tol
        
        # Top occurring sequence for each batch member
        cnt_full = counts[inds].reshape(bs, n)
        pf_full = passfail.reshape(bs, n)
        add_index = (cnt_full*pf_full).argmax(1)
        
        # Highest occurring sequence when nothing fits precursor
        all_fail = pf_full.sum(1) == 0
        add_index[all_fail] = cnt_full[all_fail].argmax(1)
        
        # Best index for every batch member
        winners = th.arange(0, full_size, n).to(device) + add_index
        assert len(winners) == bs
        top_sequences = seqs[winners]
        logits = logits[winners]
        additional_outputs = {x: y[winners] for x,y in diffout.items()}

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
        decoder_config['kv_indim'] = self.encoder.run_units
        self.decoder = MDLMDecoder(
            token_dict          = token_dict,
            decoder_config      = decoder_config,
            **decoder_config,
        )
        # Diffusion object
        self.diff_obj = MDLMDiffusion(diff_config, self.decoder.outdict, self.decoder)
        self.decoder.diff_obj = self.diff_obj

        # Scale
        if 'masses_path' in kwargs:
            path = os.path.join(kwargs['masses_path'], 'masses.tsv')
            self.str2mass = {
                m.split()[0]: float(m.split()[1]) 
                for m in open(path)
                .read().strip().split("\n")
            }
            self.int2mass = {Int: self.str2mass.get(string, 0) for string, Int in self.decoder.outdict.items()}
            self.masses = th.tensor([m[1] for m in sorted(self.int2mass.items())])
    
    def forward(self, batch, save_x=False, save_p=False, progress=False, **kwargs):
        dictionary = self.encoder_embedding(batch)
        embedding = dictionary['emb']
        spectrum_mask = dictionary['mask']
        decout = self.decoder.predict_sequence(embedding, batch, save_x=save_x, save_p=save_p, progress=progress)
        return decout

    def predict_sequence(self, batch, save_x=False, save_p=False, progress=False):
        batch_size, SL = batch['mz'].shape
        out = self(batch, save_x=save_x, save_p=save_p, progress=progress)
        return out

