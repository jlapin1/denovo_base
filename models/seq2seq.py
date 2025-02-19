import torch as th
from torch import nn
from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.diff_decoder import DenovoDiffusionDecoder
from models.decoder import DenovoDecoder
from models.diffusion.model_utils import create_diffusion
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

        #if ['encoder_name'] == 'depthcharge':
        #    self.encoder = dc_encoder(sequence_length=top_peaks)
        #else:
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
        final, logits = self.decoder.predict_sequence(embedding, batch)
        return final, logits

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
            input_output_units = diff_config['in_channel'],
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

    def forward(self, batch, save_xcur=False):
        embedding = self.encoder_embedding(batch)
        final, logits = self.decoder.predict_sequence(embedding, batch, save_xcur=save_xcur)
        return final, logits

    def predict_sequence(self, batch, save_xcur=False, n=None):
        bs, sl = batch['mz'].shape
        n = self.ens_size if n==None else n

        full_size = bs*n
        batch = self.expand_batch(batch, n=n)
        seqs, logits = self(batch, save_xcur=save_xcur)
        #uniqs, inds, counts = seqs.unique(dim=0, return_inverse=True, return_counts=True)
        
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
        
        return top_sequences, logits #counts[inds][winners]

