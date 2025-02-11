import torch as th
from torch import nn
from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.diff_decoder import DenovoDiffusionDecoder
from models.decoder import DenovoDecoder
from models.diffusion.model_utils import create_model_and_diffusion

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

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

class Seq2SeqDiff(Seq2Seq):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        diff_config,
        top_peaks,
        token_dict,
        **kwargs
    ):
        super().__init__(
            encoder_config=encoder_config,
            top_peaks=top_peaks,
        )
        decoder_config['kv_indim'] = self.encoder.run_units
        _, self.diff_obj = create_model_and_diffusion(**diff_config)
        self.decoder = DenovoDiffusionDecoder(
            input_output_units = diff_config['in_channel'],
            clip_denoised      = diff_config['clip_denoised'],
            clamp_denoised     = diff_config['clamp_denoised'],
            output_sigma       = diff_config['learn_sigma'],
            token_dict         = token_dict,
            dec_config         = decoder_config,
            diff_obj           = self.diff_obj,
            **decoder_config,
        )
    
    def forward(self, batch, save_xcur=False):
        embedding = self.encoder_embedding(batch)
        final, logits = self.decoder.predict_sequence(embedding, batch, save_xcur=save_xcur)
        return final, logits

    def predict_sequence(self, batch, save_xcur=False):
        return self(batch, save_xcur=save_xcur)

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

