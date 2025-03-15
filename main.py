"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loader import LoaderHF
import numpy as np
from models.encoder import Encoder
from models.depthcharge.SpectrumTransformerEncoder import dc_encoder
from models.heads import SequenceHead, ClassifierHead
from models.diff_decoder import DenovoDiffusionDecoder
from models.decoder import DenovoDecoder
import os
import shutil
from tqdm import tqdm
from collections import deque
from time import time
import utils as U
from copy import deepcopy
import wandb
from glob import glob
import metrics as met
import pandas as pd
nn = th.nn
F = nn.functional
choice = np.random.choice
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class BaseDenovo:
    def __init__(self, config, svdir='./downstream/'):
        
        # Config is entire downstream yaml
        self.config = config
        
        # Create directory for saving results; only use if run from PretrainModel.py
        self.log = config['save_weights']
        self.header = "HEADER"#config['header']
        if self.log and not os.path.exists(svdir):
            os.makedirs(svdir)
        if self.config['save_weights']:
            if not os.path.exists(os.path.join(svdir, 'weights')):
                os.mkdir(os.path.join(svdir, 'weights'))
        self.svdir = svdir
        self.config['sl'] = self.config['pep_length'][1]
        
        self.phase_counter = [0, 0, 0]
        if config['lr_schedule']:
            # Phase 1 warmup
            self.lr_warmup_increment = (
                (config['lr_warmup_end']-config['lr_warmup_start']) / 
                config['lr_warmup_steps']
            )
            self.starting_lr = config['lr_warmup_start']
            # Phase 2 flat
            self.lr_flat_steps = (
                eval(config['lr_flat_steps']) if type(config['lr_flat_steps']) == str else config['lr_flat_steps']
            )
            # Phase 3 decay
            lr_decay_steps = (
                eval(config['lr_decay_steps']) if type(config['lr_decay_steps']) == str else config['lr_decay_steps']
            )
            self.lr_alpha = np.exp(np.log(config['lr_floor'] / config['lr_warmup_end']) / lr_decay_steps)
            self.lr_phase = 0
        else:
            self.starting_lr = config['lr_warmup_end']
            self.lr_phase = 1
            self.config['lr_flat_steps'] = 9e9
        
        self.running_loss = []
        self.global_step = 0

        # Dataloader
        if 'val_steps' in self.config['loader'].keys(): # backwards compatibility
            val_steps = self.config['loader']['val_steps']
            self.val_steps = 1 if val_steps == None else val_steps # backwards compatiblity
        else:
            self.val_steps = 100
        self.data = LoaderHF(
            top_pks=config['top_peaks'], 
            pep_length=config['pep_length'],
            batch_size=config['batch_size'],
            **self.config['loader']
        )

        self.eval_stats = []

    def save_weights(self, fp='./model.wts'):
        th.save(self.model.state_dict(), fp)
    
    def load_saved_weights(self, obj, weights_type='model', load_last=False, retain=False):
        regex = f'*{weights_type}*last*wts*' if load_last else f"*{weights_type}*wts*"
        print(f"<DSCOMMENT> Searching for {weights_type} weights with regular expression {regex}")
        possible_weights_path = glob(os.path.join(self.svdir, "weights", regex))
        
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
            
            print(f"<DSCOMMENT> Loading {qualifier} previous {weights_type} weights: {weights_path}")
            obj.load_state_dict(th.load(weights_path, map_location=device))

            if retain:
                try:
                    shutil.copyfile(weights_path, os.path.join(self.svdir, "weights", "save"))
                except:
                    pass
        
        # Found nothing
        else:
            print(f"Found no weights fitting regular expression")

    def split_labels_str(self, incl_str):
        return [label for label in self.dl.labels if incl_str in label]

    def encinp(self, 
               batch, 
               mask_length=True, 
               return_mask=False, 
               ):

        mzab = th.cat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab.to(device),
            'charge': (
                batch['charge']
                if self.config['encoder_dict']['use_charge'] else 
                None
            ),
            'mass': (
                batch['mass']
                if self.config['encoder_dict']['use_mass'] else
                None
            ),
            'length': batch['length'] if mask_length else None,
            'return_mask': return_mask,
        }

        return model_inp
    
    def train_epoch(self, svfreq=10000):
        
        bs = self.config['batch_size']
        running_loss = {key: deque(maxlen=20) for key in self.training_loss_keys}
        
        # Progress bar
        train_steps = int(self.data.train_size // bs)
        pbar = tqdm(self.data.dataloader['train'], total=train_steps, smoothing=0.1)

        epoch_start = time()
        step_end=epoch_start
        for step, batch in enumerate(pbar):      
            step_start = time()
            
            if self.config['log_wandb']: wandb.log({"Learning rate": self.opt.param_groups[-1]['lr']})

            losses = self.train_step(batch)
            self.global_step += 1
            
            if self.config['log_wandb']:
                loss_printout = 'Loss: %7f'%losses['loss']
            else:
                for key in running_loss.keys(): running_loss[key].append(losses[key].detach().cpu())
                rlm = {key: np.mean(running_loss[key]) for key in running_loss.keys()}
                loss_printout = ", ".join(len(rlm)*['%s: %7f'])%tuple([m for n in rlm.items() for m in n])
            pbar.set_description(f"Loss: {loss_printout}")

            if self.config['log_wandb']:
                global_grad_norm = U.global_grad_norm(self.model)
                self.log_wandb(losses, global_grad_norm)
                

            self.running_loss.append(losses['loss'].detach().cpu())
            if self.log and (self.global_step % svfreq == 0):
                self.savetxt(self.running_loss)
                self.running_loss = []

            step_end = time()
            
        if self.log and (len(self.running_loss) > 0):
            self.savetxt(self.running_loss)
            self.running_loss = []
        
        print("\rFinal running loss: %s, Final time elapsed: %.0f s"%(loss_printout, time()-epoch_start))
        
    def savetxt(self, train_loss=None, eval_stats=None):
        if eval_stats is not None:
            np.savetxt(os.path.join(self.svdir, "eval_stats.txt"), np.array(eval_stats), fmt='%.6f', header=self.header)
        if train_loss is not None:
            if os.path.exists(os.path.join(self.svdir,"train_loss.txt")):
                train_loss = np.append(np.loadtxt(os.path.join(self.svdir, "train_loss.txt")), train_loss)
            np.savetxt(os.path.join(self.svdir, "train_loss.txt"), train_loss, fmt="%.6f", header=self.header)

    def update_lr(self):
        # Warmup phase
        if self.lr_phase == 0:
            if self.phase_counter[0] < self.config['lr_warmup_steps']:
                self.opt.param_groups[-1]['lr'] += self.lr_warmup_increment
                self.phase_counter[0] += 1
            else:
                self.lr_phase = 1
        # Flat phase
        elif self.lr_phase == 1:
            if self.phase_counter[1] < self.lr_flat_steps:
                self.phase_counter[1] += 1
            else:
                self.lr_phase = 2
        # Decay phase
        else:
            if self.opt.param_groups[-1]['lr'] > self.config['lr_floor']:
                lr = self.config['lr_warmup_end']*self.lr_alpha**self.phase_counter[2]
                self.opt.param_groups[-1]['lr'] = lr
                self.phase_counter[2] += 1
            else:
                self.opt.param_groups[-1]['lr'] = self.config['lr_floor']

    def replace_with_eos_token(self, intseq, lengths):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.model.decoder.EOS

        return intseq

    def append_null_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.model.decoder.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def fill_null_after_first_eos_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        length = ((intseq == self.model.decoder.EOS)|(intseq == self.model.decoder.NT)).int().argmax(1)
        intseq[th.arange(bs), length] = self.model.decoder.EOS
        mask = (length > 0)[:,None]
        index_array = th.arange(sl)[None].repeat([bs, 1]).to(intseq.device)
        boolean_array = index_array > length[:, None]
        intseq[mask & boolean_array] = self.model.decoder.NT

        return intseq

    def to_list_of_strings(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        return [
            [
                self.model.decoder.rev_outdict[int(n)] 
                for n in m if n not in [self.model.decoder.NT, self.model.decoder.EOS]
            ] 
            for m in intseq
        ]

    def evaluation(self, dset='val', max_batches=1e10, save_df=False, kwargs={}):
        
        # Dataframe
        if save_df:
            dataframe = {
                'targ_intseq': [],
                'charge': [],
                'mass': [],
                'peptide_length': [],
                'pred_intseq': [],
                'probs': [],
                'targ_aaseq': [],
                'pred_aaseq': [],
                'correct_aa': [],
                'correct_peptide': [],
            }

        # losses
        out = {'ce': 0}
        tots = {'sum':{}, 'total': {}}

        # Progress bar
        val_steps = min(
            self.data.val_size // self.data.dataloader[dset].batch_size,
            max_batches,
        )
        pbar = tqdm(self.data.dataloader[dset], total=val_steps, leave=False)
        
        self.model.eval()
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Evaluation")
            if i == max_batches:
                break

            #print("\rEvaluation step %d"%(i+1), end='')
            batchdev = U.Dict2dev(batch, device)
            with th.no_grad():
                seqint, target, loss_mask = self.inptarg(batchdev)
                prediction, probs = self.model.predict_sequence(batchdev, **kwargs)
            
            # Do some resizing/reshaping
            prediction = prediction[..., :target.shape[1]] # loaded shapes can change based on batch
            probs = probs[:, :target.shape[1]]
            predicted_probs = probs.softmax(-1).gather(-1, prediction[...,None].type(th.int64)).squeeze()
            pred = probs.transpose(-1,-2)
            
            # Cross entropy
            out['ce'] += (
                F.cross_entropy(pred, target, reduction='none')[loss_mask].sum()
            )
            
            # Deepnovo metrics
            prediction = self.fill_null_after_first_eos_token(prediction)
            pred_strings = self.to_list_of_strings(prediction)
            targ_strings = self.to_list_of_strings(target)
            aa_matches_batch, n_aa1, n_aa2 = met.aa_match_batch(pred_strings, targ_strings, self.data.massdic)
            dn_metrics = {
                'sum': {
                    'aa_recall': sum([sum(m[0]) for m in aa_matches_batch]),
                    'aa_precision': sum([sum(m[0]) for m in aa_matches_batch]),
                    'peptide': sum([m[-1] for m in aa_matches_batch]),
                },
                'total': {
                    'aa_recall': n_aa2,
                    'aa_precision': n_aa1,
                    'peptide': len(aa_matches_batch),
                },
            }

            if save_df:
                dataframe['charge'].extend(batch['charge'].cpu().numpy().tolist())
                dataframe['mass'].extend(batch['mass'].cpu().numpy().tolist())
                dataframe['peptide_length'].extend(batch['peplen'].cpu().numpy().tolist())
                dataframe['targ_intseq'].extend(batch['intseq'].cpu().numpy().tolist())
                dataframe['pred_intseq'].extend(prediction.cpu().numpy().tolist())
                dataframe['probs'].extend(predicted_probs.cpu().numpy().tolist())
                dataframe['targ_aaseq'].extend(targ_strings)
                dataframe['pred_aaseq'].extend(pred_strings)
                dataframe['correct_aa'].extend([result[0] for result in aa_matches_batch])
                dataframe['correct_peptide'].extend([result[1] for result in aa_matches_batch])
            
            # Add to totals
            for metric in dn_metrics['sum'].keys():
                if metric not in tots['sum'].keys():
                    tots['sum'][metric] = 0
                    tots['total'][metric] = 0
                tots['sum'][metric] += dn_metrics['sum'][metric]
                tots['total'][metric] += dn_metrics['total'][metric]

            self.on_eval_step_end(target, loss_mask)
        
        steps = i+1
        totsz = self.config['batch_size']*steps
        out['ce'] = float((out['ce'] / (totsz * self.config['sl'])).cpu().detach().numpy())
        for metric in tots['sum'].keys():
            out[metric] = tots['sum'][metric] /  tots['total'][metric]

        self.on_eval_end()
        
        if save_df:
            return out, pd.DataFrame(dataframe)
        else:
            return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        highscore = 0
        for i in range(self.config['epochs']):
            
            # Train
            self.data.dataset['train'].set_epoch(i)
            self.train_epoch()
            self.on_train_epoch_end()
            
            # Eval
            out = self.evaluation(dset=eval_dset, max_batches=self.val_steps, kwargs=self.eval_kwargs)
            
            # Logging
            if self.config['log_wandb']:
                out['epoch'] = i+1
                wandb.log(out)
                out.pop('epoch')
            
            specifier = " ".join(len(out)*['%s'])
            write_out = specifier%tuple([f"{m}={n:.3}" for m,n, in out.items()])
            line = "ValEpoch %d: %s"%(i, write_out)
            
            if out[self.config['high_score']] > highscore:
                highline = line
                highscore = out[self.config['high_score']]
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print("\r"+line)
            
            # Saving the checkpoint
            if self.config['save_weights']:
                self.save_weights(os.path.join(self.svdir, 'weights/model_last.wts'))
                U.save_optimizer_state(self.opt, os.path.join(self.svdir, 'weights/opt_last.wts'))
                if highscore == out[self.config['high_score']]:
                    ext = f"epoch{i}_high_{highscore:.3f}"
                    wtsdir = os.path.join(self.svdir, "weights")
                    for file in glob(os.path.join(wtsdir, "*high*")): os.remove(file)
                    self.save_weights(os.path.join(wtsdir, f"model_{ext}.wts"))
            
            self.eval_stats.append(list(out.values()))
            
            # Save data
            if self.log:
                self.savetxt(train_loss=None, eval_stats=np.array(self.eval_stats))
            
        return lines, highline

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_eval_step_end(self, *args, **kwargs):
        pass

    def on_eval_end(self, *args, **kwargs):
        pass


class DenovoArDSObj(BaseDenovo):
    def __init__(self, config, svdir='./dswts/'):
        super().__init__(
            config=config,
            svdir=svdir
        )
        self.training_loss_keys = ['loss']
        self.eval_kwargs = {}
        
        from models.seq2seq import Seq2SeqAR
        
        self.model = Seq2SeqAR(
            encoder_config = config['encoder_dict'],
            decoder_config = config['decoder_ar'],
            token_dict     = self.data.amod_dic,
            top_peaks      = config['top_peaks'],
        )

        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        self.predict_sequence = self.model.decoder.predict_sequence
        
        # loading previous weights
        if config['prev_wts'] is not None:
            retain = False if config['load_last'] else True
            self.load_saved_weights(self.model, "model", config['load_last'], retain=retain)
            self.load_saved_weights(self.opt, "opt", config['load_last'])     
            U.optimizer_to(self.opt, device)

        self.model.to(device)
    
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])
        
        dec_input = self.model.decoder.prepend_startok(dec_input)

        target = self.append_null_token(target)
        target = self.replace_with_eos_token(target, batch['peplen'])

        loss_mask = self.model.decoder.sequence_mask(batch['peplen'], target.shape[1])
        loss_mask = loss_mask == 0

        return dec_input, target, loss_mask

    def LossFunction(self, target, prediction, loss_mask):
        targ_one_hot = F.one_hot(target, self.model.decoder.predcats).type(th.float32)
        targ_one_hot = targ_one_hot.transpose(-1,-2)
        prediction = prediction.transpose(-1,-2)
        all_loss = F.cross_entropy(prediction, targ_one_hot, reduction='none')
        # Consider also mean of instances rather than mean of all tokens
        masked_loss = all_loss[loss_mask]
        loss = masked_loss.sum() / loss_mask.sum()

        return loss

    def train_step(self, batch, trenc=True):
        batch = U.Dict2dev(batch, device)
        #enc_input, seqint, target = self.inptarg(batch)
        dec_input, target, loss_mask = self.inptarg(batch)
        
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()
        logits = self.model(dec_input, batch)
        all_loss = self.LossFunction(target, logits, loss_mask)
        loss = all_loss.mean()
        
        loss.backward()
        
        self.update_lr()
        self.opt.step()
        
        return {'loss': loss}

    def log_wandb(self, losses, norm):
        wandb.log({
            "Total loss": losses['loss'],
            'Global step': self.global_step,
            "Global grad norm": norm,
        })

class DenovoDiffusionObj(BaseDenovo):
    def __init__(self, config, diff_config=None, svdir='./dswts/'):
        super().__init__(
            config=config, 
            svdir=svdir
        )
        self.training_loss_keys = ['loss', 'mse', 'decoder_nll', 'tT']
        self.eval_kwargs = {'n': 1}

        # Diffusion object
        if config['decoder_diff']['diffusion_config']['learn_sigma']: 
            self.training_loss_keys.append("vlb_terms")
        config['decoder_diff']['diffusion_config']['pad_tok_id'] = self.data.amod_dic['X']
        config['decoder_diff']['diffusion_config']['resume_checkpoint'] = False
        config['decoder_diff']['diffusion_config']['sequence_len'] = self.config['pep_length'][1] + 1 # b/c of eos token
        self.diff_config = config['decoder_diff']['diffusion_config']
        
        from models.seq2seq import Seq2SeqDiff

        # diffusion object created inside Seq2Seq
        self.model = Seq2SeqDiff(
            encoder_config  = config['encoder_dict'], 
            decoder_config  = config['decoder_diff']['model_config'], 
            diff_config     = config['decoder_diff']['diffusion_config'],
            ensemble_config = config['decoder_diff']['ensemble'], 
            top_peaks = config['top_peaks'], 
            max_peptide_length = config['pep_length'][1], 
            token_dict = self.data.amod_dic,
            masses_path = config['loader']['masses_path'],
        )
        
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)
        
        # loading previous weights
        if config['prev_wts'] is not None:
            retain = False if config['load_last'] else True
            self.load_saved_weights(self.model, "model", config['load_last'], retain=retain)
            self.load_saved_weights(self.opt, "opt", config['load_last'])
            U.optimizer_to(self.opt, device)
        
        self.model.to(device)
        
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])

        # Schedule sampler
        timesteps = th.empty(bs).uniform_(
            0, self.model.diff_obj.num_timesteps
        ).floor().type(th.int32).to(target.device)
        
        target = self.model.decoder.append_null_token(target)
        target = self.model.decoder.replace_with_eos_token(target, batch['peplen'])

        loss_mask = self.model.decoder.sequence_mask(target)

        return timesteps, target, loss_mask

    def train_step(self, batch):
        batch = U.Dict2dev(batch, device)
        timesteps, target, loss_mask = self.inptarg(batch)

        self.model.to(device)
        self.model.train()
        self.model.zero_grad()
        
        embedding = self.model.encoder_embedding(batch)
        
        model_kwargs = {
            'input_ids': None,
            'decoder_input_ids': target,
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_feats': embedding['emb'],
        }
        if self.diff_config['use_loss_mask']:
            model_kwargs['loss_mask'] = loss_mask # THIS RUINS EVERYTHING

        losses = self.model.diff_obj.training_losses(
            self.model.decoder, 
            self.global_step,
            timesteps, 
            model_kwargs=model_kwargs, 
            noise=None
        )
        
        losses = {key: loss.mean() for key, loss in losses.items()}
        loss = losses['loss']
        loss.backward()
        
        self.update_lr()
        self.opt.step()
        
        return losses
    
    def log_wandb(self, losses, grad_norm):
        wandb.log({
            "Total loss": losses['loss'],
            "MSE loss": losses['mse'],
            "DecoderNLL loss": losses['decoder_nll'],
            "tT loss": losses['tT'],
            'Global step': self.global_step,
            "Global grad norm": grad_norm,
        })
        if 'vlb_terms' in losses:
            wandb.log({'VLB loss': losses['vlb_terms'],})
   
    def on_train_epoch_end(self):
        avg_losses = self.model.diff_obj.my_loss_history / (self.model.diff_obj.my_loss_count+1e-7)[...,None]
        save_path = os.path.join(self.svdir, "train_loss_by_timestep.tab")
        np.savetxt(save_path, avg_losses, delimiter='\t', fmt='%.8f')
        self.model.diff_obj.my_loss_history = np.zeros((self.model.diff_obj.num_timesteps, 3))
        self.model.diff_obj.my_loss_count = np.zeros((self.model.diff_obj.num_timesteps,))
    
    def on_eval_step_end(self, target, mask):
        pass

    def on_eval_end(self):
        pass

if __name__ == '__main__':

    # Read yamls
    with open("./yaml/config.yaml") as stream:
        config = yaml.safe_load(stream)
    # Overrides over a loaded previous experiment
    config_ = config.copy()
    # Eval config will not be overwritten
    with open("./yaml/eval.yaml") as stream:
        evconfig = yaml.safe_load(stream)

    ########################################################
    # Create experiment directory in save/downstream_only/ #
    ########################################################

    # Continuing previous downstream run
    if config['prev_wts'] is not None:
        svdir = os.path.join(config['prev_wts'])
        with open(os.path.join(config['prev_wts'], "yaml", "config.yaml")) as stream:
            config = yaml.safe_load(stream)
        # Replace previous settings with new ones
        for key in [
            'epochs', 'prev_wts', 'load_last', 'lr', 'lr_warmup', 
            'lr_warmup_start', 'lr_warmup_end', 'lr_warmup_steps',
            'loader', 'log_wandb', 'eval_only', 'batch_size',
            'top_peaks',
        ]:
            config[key] = config_[key]
        timestamp = config['prev_wts']
    # Create new experiment
    elif config['save_weights']:
        timestamp = U.timestamp()
        svdir = os.path.join('save', timestamp)
        U.create_experiment(svdir, svwts=config['save_weights'])
        print("<DSCOMMENT> Experiment is writing to directory %s"%svdir)
    else:
        svdir = './'
        
    # Downstream object
    print("<DSCOMMENT> Denovo sequencing")
    if 'diff' in config['decoder_name']:
        print("<DSCOMMENT> Using diffusion decoder")
        D = DenovoDiffusionObj(config, svdir=svdir)
    else:
        print("<DSCOMMENT> Using autoregressive decoder")
        D = DenovoArDSObj(config, svdir=svdir)

    # WandB
    if config['log_wandb'] and (config['eval_only'] == False):
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
			config={
				'master': config,
                'save_directory': timestamp,
                'model_parameters': D.model.total_params(),
			},
		)   

    # Run training and/or evaluation
    if config['eval_only']:
        evc = evconfig['eval_only']
        
        # Apply settings that are independent of training
        max_batches = int(eval(str(evc['max_batches'] if evc['max_batches'] is not None else 9e10)))
        if 'diff' in config['decoder_name']:
            if evc['clamp_denoised'] is not None:
                D.model.decoder.clamp_denoised = evc['clamp_denoised']
            if evc['n'] is not None:
                D.model.ens_size = evc['n']

        # Run evaluation
        out, df = D.evaluation(dset=evc['set'], max_batches=max_batches, save_df=True)
        
        # Saving results
        if evc['save']:
            eval_out_path = evc['outpath'] if evc['outpath'] is not None else os.path.join(svdir, "output.parquet")
            df.to_parquet(eval_out_path)
        print("\n", out)
    else:
        print("Test validation", end='')
        out = D.evaluation(dset='val', max_batches=2, kwargs=D.eval_kwargs)
        assert D.config['high_score'] in out.keys()
        print("\rTest validation passed")
        print(D.TrainEval()[-1])
