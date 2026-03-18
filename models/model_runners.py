import torch as th
from loader import LoaderHF
import numpy as np
from models.diff_classifier import Classifier
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
import pyarrow as pa
import pyarrow.parquet as pq
import math
F = th.nn.functional
device = th.device("cuda" if th.cuda.is_available() else "cpu")

class BaseDenovo:
    def __init__(self, config, svdir='./downstream/', rddir=None):
        
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
        self.rddir = rddir
        self.config['sl'] = self.config['pep_length'][1]
        
        self.phase_counter = [0, 0, 0]
        if config['lr_schedule']:
            # Phase 1 warmup
            self.lr_warmup_increment = (
                (config['lr_warmup_end']-config['lr_warmup_start']) / 
                np.maximum(config['lr_warmup_steps'], 1)
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
            self.starting_lr = eval(config['lr_warmup_end']) if type(config['lr_warmup_end']) == str else config['lr_warmup_end']
            self.lr_phase = 1
            self.lr_flat_steps = 9e9
        
        self.running_loss = []
        self.global_step = 0
        self.save_last_counter = 0

        self.high_score = 0
        self.eval_frequency = config['eval_frequency']
        self.eval_time = time()

        # Dataloader
        if 'val_steps' in self.config['loader'].keys(): # backwards compatibility
            val_steps = self.config['loader']['val_steps']
            self.val_steps = 1 if val_steps == None else val_steps # backwards compatiblity
        else:
            self.val_steps = 100
        self.reverse = config['loader']['reverse']
        self.data = LoaderHF(
            top_pks=config['top_peaks'], 
            pep_length=None if (config['inference']&config['eval_only']) else config['pep_length'],
            batch_size=config['batch_size'],
            **self.config['loader']
        )
        
        self.training_loss_keys = []
        self.eval_stats = []
        self.eval_kwargs = {}

    def save_weights(self, fp='./model.wts'):
        th.save(self.model.state_dict(), fp)
    
    def save_last(self, override=False):
        ready = self.global_step - self.save_last_counter >= self.config['save_last_freq']
        if ready or override:
            self.save_weights(os.path.join(self.svdir, 'weights/model_last.wts'))
            U.save_optimizer_state(self.opt, os.path.join(self.svdir, 'weights/opt_last.wts'))
            self.save_last_counter = self.global_step
    
    def checkpoint(self, score):
        self.save_last(override=True)
        if score > self.high_score:
            self.high_score = score
            ext = f"step{self.global_step}_high_{self.high_score:.3f}"
            wtsdir = os.path.join(self.svdir, "weights")
            for file in glob(os.path.join(wtsdir, "*high*")): os.remove(file)
            self.save_weights(os.path.join(wtsdir, f"model_{ext}.wts"))

    def load_saved_weights(self, obj, weights_type='model', load_last=False, retain=False):
        regex = f'*{weights_type}*last*wts*' if load_last else f"*{weights_type}*wts*"
        print(f"<DSCOMMENT> Searching for {weights_type} weights with regular expression {regex}")
        possible_weights_path = glob(os.path.join(self.rddir, "weights", regex))
        
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
            obj.load_state_dict(th.load(weights_path, map_location=device, weights_only=False))

            if retain:
                try:
                    shutil.copyfile(weights_path, os.path.join(self.svdir, "weights", "save"))
                except:
                    pass
        
        # Found nothing
        else:
            print(f"Found no weights fitting regular expression")
    
    def restore_model(self):
        if self.config['prev_wts'] is not None:
            retain = False if self.config['load_last'] else True
            self.load_saved_weights(self.model, "model", self.config['load_last'], retain=retain)
            if self.config['load_last']:
                self.load_saved_weights(self.opt, "opt", self.config['load_last'])
                U.optimizer_to(self.opt, device)

    def split_labels_str(self, incl_str):
        return [label for label in self.dl.labels if incl_str in label]
    
    def train_epoch(self, svfreq=10000):
        
        bs = self.config['batch_size']
        running_loss = {key: deque(maxlen=20) for key in self.training_loss_keys}
        
        # Progress bar
        train_steps = int(self.data.train_size // bs)
        pbar = tqdm(self.data.dataloader['train'], total=train_steps, smoothing=0.6)

        epoch_start = time()
        step_end=epoch_start
        for step, batch in enumerate(pbar):      
            step_start = time()
            
            if self.config['log_wandb']: wandb.log({"Learning rate": self.opt.param_groups[-1]['lr']})
            
            losses = self.train_step(batch)
            total_loss = losses['loss']
            self.global_step += 1
            
            if self.config['log_wandb']:
                loss_printout = 'Loss: %7f'%losses['loss']
                global_grad_norm = U.global_grad_norm(self.model)
                self.log_wandb(losses, global_grad_norm)
            else:
                for key in running_loss.keys(): running_loss[key].append(losses[key])
                rlm = {key: np.mean(running_loss[key]) for key in running_loss.keys()}
                loss_printout = ", ".join(len(rlm)*['%s: %7f'])%tuple([m for n in rlm.items() for m in n])
            pbar.set_description(f"Loss: {loss_printout}")
            
            self.running_loss.append(total_loss)
            if self.log and (self.global_step % svfreq == 0):
                self.savetxt(self.running_loss)
                self.running_loss = []
            if self.config['save_weights']:
                self.save_last()
            if self.eval_frequency is not None and (self.config['save_weights']|self.config['log_wandb']):
                if time()-self.eval_time > self.eval_frequency:
                    out, _ = self.evaluation(dset='val', max_batches=self.val_steps, kwargs=self.eval_kwargs)
                    self.eval_out = out
                    new_score = out[self.config['high_score']]
                    if self.config['save_weights']: self.checkpoint(new_score)
                    if self.config['log_wandb']: wandb.log(out)
                    self.eval_time = time()
            
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
                self.opt.param_groups[-1]['lr'] = self.config['lr_warmup_end'] # Notig fur einen Neustart
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
        eos_inds = (th.arange(bs, device=intseq.device), lengths)
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
        is_reverse = lambda x: x[::-1] if self.reverse else x
        return [
            is_reverse([
                self.model.decoder.rev_outdict[int(n)] 
                for n in m if n not in [self.model.decoder.NT, self.model.decoder.EOS]
            ])
            for m in intseq
        ]

    def inference(
        self,
        output_filename="./hold.parquet",
        dset='val',
        max_batches=1e10,
        stream_write=False,
        batches_btw_write=10,
        no_grad=True,
        kwargs={},
        save_keys=[],
    ):
        schema_defined = False

        def initial_dataframe(additional_keys: list=[]):
            out = {
                #'pred_intseq': [], 
                'pred_aaseq': [], 
                'probs': []
            }
            return out
        dataframe = initial_dataframe()
        
        # Progress bar
        val_steps = min(
            self.data.val_size // self.data.dataloader[dset].batch_size,
            max_batches,
        )
        pbar = tqdm(self.data.dataloader[dset], total=val_steps, leave=True)
        pbar.set_description(f"Inference")
        self.model.eval()
        for i, batch in enumerate(pbar):
            
            if i == max_batches:
                break

            #print("\rEvaluation step %d"%(i+1), end='')
            batchdev = U.Dict2dev(batch, device)
            if no_grad:
                with th.no_grad():
                    #seqint, target, loss_mask = self.inptarg(batchdev)
                    out_dict = self.model.predict_sequence(batchdev, **kwargs)
            else:
                #seqint, target, loss_mask = self.inptarg(batchdev)
                out_dict = self.model.predict_sequence(batchdev, **kwargs)
            
            # process prediction
            prediction = out_dict.pop('prediction')
            prediction = self.fill_null_after_first_eos_token(prediction)
            pred_strings = self.to_list_of_strings(prediction)
            
            # Process lgoits
            probs = out_dict.pop('logits')
            predicted_probs = probs.softmax(-1).gather(-1, prediction[...,None]).squeeze()
            
            # Collect results
            #dataframe['pred_intseq'].extend(prediction.cpu().numpy().tolist())
            dataframe['pred_aaseq'].extend(pred_strings)
            dataframe['probs'].extend(predicted_probs.cpu().numpy())
            combined_dict = batch | out_dict
            for key in save_keys:
                value = combined_dict[key]
                valuetype = type(value)
                if key not in dataframe:
                    dataframe[key] = []
                if valuetype == list:
                    dataframe[key].extend(value)
                elif valuetype == np.ndarray:
                    dataframe[key].extend(value)
                elif valuetype == th.Tensor:
                    dataframe[key].extend(value.cpu().numpy())
                else:
                    dataframe[key].extend(value)
                
            # Prevent error at the end of evaluation when not streaming
            length_first = len(dataframe['pred_aaseq'])
            array = np.array([len(value) for key, value in dataframe.items()])
            assert (length_first == array).all(), f"batch#: {i}, {dataframe.keys()}, {array}"
            
            if stream_write and ((i+1) % batches_btw_write == 0):
                dataframe_ = {key: value for key, value in dataframe.items() if len(value)>0}   
                table = pa.Table.from_pandas(pd.DataFrame(dataframe_), preserve_index=False)
                if not schema_defined:
                    writer = pq.ParquetWriter(f'{output_filename}', table.schema, compression='snappy')
                    schema_defined = True
                writer.write_table(table)
                dataframe = initial_dataframe()
        writer.close()

    def evaluation(
        self, 
        dset='val', 
        max_batches=1e10, 
        save_df=False,
        stream_write=False,
        batches_btw_write=10,
        no_grad=True, 
        kwargs={}
    ):
        
        # Dataframe
        def initial_dataframe(extra_keys: list=[]):
            dataframe = {
                'name': [],
                #'chimeric': [],
                #'hyperscore': [],
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
            for key in extra_keys: dataframe[key] = []
            return dataframe
        if save_df or stream_write:
            dataframe = initial_dataframe()
            schema_defined = False
        else:
            dataframe = None

        # losses
        out = {'ce': 0}
        tots = {'sum':{}, 'total': {}}

        # Progress bar
        val_steps = min(
            self.data.val_size // self.data.dataloader[dset].batch_size,
            max_batches,
        )
        pbar = tqdm(self.data.dataloader[dset], total=val_steps, leave=False)
        pbar.set_description(f"Evaluation")
        self.model.eval()
        for i, batch in enumerate(pbar):
            
            if i == max_batches:
                break

            #print("\rEvaluation step %d"%(i+1), end='')
            batchdev = U.Dict2dev(batch, device)
            if no_grad:
                with th.no_grad():
                    seqint, target, loss_mask = self.inptarg(batchdev)
                    out_dict = self.model.predict_sequence(batchdev, **kwargs)
            else:
                seqint, target, loss_mask = self.inptarg(batchdev)
                out_dict = self.model.predict_sequence(batchdev, **kwargs)
            prediction = out_dict.pop('prediction')
            probs = out_dict.pop('logits')
            
            # Do some resizing/reshaping
            prediction = prediction[..., :target.shape[1]] # loaded shapes can change based on batch
            probs = probs[:, :target.shape[1]]
            predicted_probs = probs.softmax(-1).gather(-1, prediction[...,None].type(th.int64)).squeeze(-1)
            
            # Cross entropy
            pred = probs.transpose(-1,-2)
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

            # Add to totals
            for metric in dn_metrics['sum'].keys():
                if metric not in tots['sum'].keys():
                    tots['sum'][metric] = 0
                    tots['total'][metric] = 0
                tots['sum'][metric] += dn_metrics['sum'][metric]
                tots['total'][metric] += dn_metrics['total'][metric]
            
            if save_df or stream_write:
                self.on_eval_step_end(batchdev, out_dict, dataframe)
                dataframe['name'].extend(batch['experiment_name'])
                if 'chimeric' in batch:
                    if 'chimeric' not in dataframe: dataframe['chimeric'] = []
                    dataframe['chimeric'].extend(batch['chimeric'].cpu().numpy().tolist())
                if 'hyperscore' in batch:
                    if 'hyperscore' not in dataframe: dataframe['hyperscore'] = []
                    dataframe['hyperscore'].extend(batch['hyperscore'].cpu().numpy().tolist())
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
                
                # Prevent error at the end of evaluation when not streaming
                length_first = len(dataframe['name'])
                array = np.array([len(value) for key, value in dataframe.items()])
                assert (length_first == array).all(), f"batch#: {i}, {dataframe.keys()}, {array}"
				
                if stream_write and ((i+1) % batches_btw_write == 0):
                    dataframe_ = {key: value for key, value in dataframe.items() if len(value)>0}   
                    table = pa.Table.from_pandas(pd.DataFrame(dataframe_), preserve_index=False)
                    if not schema_defined:
                        writer = pq.ParquetWriter('./hold.parquet', table.schema, compression='snappy')
                        schema_defined = True
                    writer.write_table(table)
                    dataframe = initial_dataframe()

            # Add to totals
            for metric in dn_metrics['sum'].keys():
                if metric not in tots['sum'].keys():
                    tots['sum'][metric] = 0
                    tots['total'][metric] = 0
                tots['sum'][metric] += dn_metrics['sum'][metric]
                tots['total'][metric] += dn_metrics['total'][metric]

            #self.on_eval_step_end(target, loss_mask, dataframe=dataframe)
        
        steps = i+1
        totsz = self.config['batch_size']*steps
        out['ce'] = float((out['ce'] / (totsz * self.config['sl'])).cpu().detach().numpy())
        for metric in tots['sum'].keys():
            out[metric] = float(tots['sum'][metric] /  tots['total'][metric])
        
        self.on_eval_end()
        
        if stream_write and (len(dataframe['name']) > 0):
            table = pa.Table.from_pandas(pd.DataFrame(dataframe), preserve_index=False)
            writer.write_table(table)
            writer.close()
            return out, None
        elif save_df:   
            return out, pd.DataFrame(dataframe)
        else:
            return out, None

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        for i in range(self.config['epochs']):
            
            # Train
            self.data.dataset['train'].set_epoch(i)
            self.train_epoch()
            self.on_train_epoch_end()
            
            # Eval
            if self.eval_frequency is None:
                out, _ = self.evaluation(dset=eval_dset, max_batches=self.val_steps, kwargs=self.eval_kwargs)
                new_score = out[self.config["high_score"]]
                if self.config['save_weights']: self.checkpoint(new_score)
            
                # Logging
                if self.config['log_wandb']:
                    out['epoch'] = i+1
                    wandb.log(out)
                    out.pop('epoch')
            else:
                out = self.eval_out if hasattr(self, 'eval_out') else self.evaluation(dset=eval_dset, max_batches=self.val_steps, kwargs=self.eval_kwargs)[0]
                new_score = out[self.config["high_score"]]
            
            specifier = " ".join(len(out)*['%s'])
            write_out = specifier%tuple([f"{m}={n:.3}" for m,n, in out.items()])
            line = "ValEpoch %d: %s"%(i, write_out)
            
            if new_score > self.high_score:
                highline = line
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print("\r"+line)
            
            # Saving the checkpoint
            #if self.config['save_weights']:
            #    self.checkpoint(new_score)
                #self.save_last(override=True)
                #if self.high_score == new_score:
                #    ext = f"epoch{i}_high_{self.high_score:.3f}"
                #    wtsdir = os.path.join(self.svdir, "weights")
                #    for file in glob(os.path.join(wtsdir, "*high*")): os.remove(file)
                #    self.save_weights(os.path.join(wtsdir, f"model_{ext}.wts"))
            
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
    def __init__(self, config, svdir='./dswts/', rddir=None):
        super().__init__(
            config=config,
            svdir=svdir,
            rddir=rddir,
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
        
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")

        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        self.predict_sequence = self.model.decoder.predict_sequence
        
        # loading previous weights
        self.restore_model()
                
        self.model.to(device)
    
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])
        
        dec_input = self.model.decoder.prepend_startok(dec_input)
        
        #batch['mass'] = batch['mass'] * batch['charge'] # for MKB trained model

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
        
        return {'loss': loss.item()}

    def log_wandb(self, losses, norm):
        wandb.log({
            "Total loss": losses['loss'],
            'Global step': self.global_step,
            "Global grad norm": norm,
        })

class DenovoDiffusionObj(BaseDenovo):
    def __init__(self, config, diff_config=None, svdir='./dswts/', rddir=None):
        super().__init__(
            config=config, 
            svdir=svdir,
            rddir=rddir,
        )
        self.training_loss_keys = ['loss', 'mse', 'decoder_nll', 'tT']
        self.eval_kwargs = {'n': config['decoder_diff']['ensemble']['ensemble_n']}

        # Diffusion object
        if config['decoder_diff']['diffusion_config']['learn_sigma']: 
            self.training_loss_keys.append("vlb_terms")
        config['decoder_diff']['diffusion_config']['pad_tok_id'] = self.data.amod_dic['X']
        config['decoder_diff']['diffusion_config']['resume_checkpoint'] = False
        config['decoder_diff']['diffusion_config']['sequence_len'] = self.config['pep_length'][1] + 1 # b/c of eos token
        config['decoder_diff']['model_config']['sequence_length'] = self.config['pep_length'][1] + 1
        self.diff_config = config['decoder_diff']['diffusion_config']

        from models.seq2seq import Seq2SeqDiff

        # diffusion object created inside Seq2Seq
        self.model = Seq2SeqDiff(
            encoder_config    = config['encoder_dict'], 
            decoder_config    = config['decoder_diff']['model_config'], 
            diff_config       = config['decoder_diff']['diffusion_config'],
            ensemble_config   = config['decoder_diff']['ensemble'],

            top_peaks = config['top_peaks'], 
            max_peptide_length = config['pep_length'][1], 
            token_dict = self.data.amod_dic,
            masses_path = config['loader']['masses_path'],
        )

        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)
        
        # loading previous weights
        self.restore_model()
                
        self.model.to(device)

        # Classifier
        classifier_config = config['classifier_config']
        if classifier_config['ckpt']:
            classifier_config['diffdir'] = config['prev_wts']
            self.classifier = Classifier(
                classifier_config['diffdir'],
                num_input_tokens=len(self.model.decoder.outdict),
                num_output_classes=classifier_config['num_output_classes'],
                null_token=self.model.decoder.NT,
            )
            self.classifier.load_weights(classifier_config['ckpt'])
            self.classifier.eval()
            self.classifier.to(device)
            self.eval_kwargs['cls_dict'] = {
                'model': self.classifier, 
                'index': classifier_config['class_index'], 
                'scale': classifier_config['scale'],
            }
        
    def inptarg(self, batch):
        
        bs, sl = batch['intseq'].shape
        #dec_input = deepcopy(batch['intseq'])
        target = deepcopy(batch['intseq'])

        #batch['mass'] = batch['mass'] * batch['charge'] # For MKB trained model

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
        
        loss = losses['loss'].mean()
        loss.backward()
        with th.no_grad():
            losses = {key: loss.mean().detach().cpu().item() for key, loss in losses.items()}
        
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
        try:
            avg_losses = self.model.diff_obj.my_loss_history / (self.model.diff_obj.my_loss_count+1e-7)[...,None]
            save_path = os.path.join(self.svdir, "train_loss_by_timestep.tab")
            np.savetxt(save_path, avg_losses, delimiter='\t', fmt='%.8f')
            self.model.diff_obj.my_loss_history = np.zeros((self.model.diff_obj.num_timesteps, 3))
            self.model.diff_obj.my_loss_count = np.zeros((self.model.diff_obj.num_timesteps,))
        except:
            pass
    
    def on_eval_step_end(self, batch, out_dict, dataframe=None):
        if dataframe is None:
            return 0
        if 'entropy' not in dataframe:
            dataframe['entropy'] = []
        
        entropies = out_dict['entropy']
        pl = batch['peplen']
        bs, sl = entropies.shape
        mask = th.arange(sl, device=pl.device)[None].tile([bs, 1]) <= pl[:,None]
        peptide_entropies = (entropies*mask).sum(1) / pl
        dataframe['entropy'].extend(peptide_entropies.cpu().numpy().tolist())

    def on_eval_end(self):
        pass

class DenovoMDLMObj(BaseDenovo):
    def __init__(self, config, svdir='./save/', rddir=None):
        super().__init__(
            config=config, 
            svdir=svdir,
            rddir=rddir,
        )
        self.training_loss_keys.extend(['loss'])
        self.eval_kwargs = {}

        from models.seq2seq import Seq2SeqMDLM
        
        diff_config = config['decoder_mdlm']['diffusion_config']
        self.diff_config = diff_config
        self.max_length = diff_config['model']['length']
        self.steps = diff_config['sampling']['steps']
        config['decoder_diff']['diffusion_config']['pad_tok_id'] = self.data.amod_dic['X']
        config['decoder_diff']['diffusion_config']['resume_checkpoint'] = False
        config['decoder_diff']['diffusion_config']['sequence_len'] = self.config['pep_length'][1] + 1 # b/c of eos token
        
        # The diffusion models share the model_config, but with a la carte alterations
        config['decoder_diff']['model_config']['self_condition'] = diff_config['model']['self_condition']
        
        self.model = Seq2SeqMDLM(
            encoder_config = config['encoder_dict'],
            decoder_config = config['decoder_diff']['model_config'],
            diff_config = diff_config,
            top_peaks = config['top_peaks'], 
            max_peptide_length = config['pep_length'][1], 
            token_dict = self.data.amod_dic,
            ensemble_config   = config['decoder_diff']['ensemble'],
            masses_path = config['loader']['masses_path'],
        )
        self.initialize_token_loss()
                
        # Optimizer
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)
        
        # loading previous weights
        self.restore_model()
        
        self.model.to(device)

        self.weightmat = lambda length, p=0.1: (math.log(1-p)*((th.arange(length)[None]-th.arange(length)[:,None]).abs()-1) + math.log(p)).exp() * 0.5 * (th.eye(length)==0).float()
        self.BlockMasks = lambda typ, sequence_length, block_size, precursor_token=False: U.BlockMasks(typ, sequence_length, block_size, precursor_token).to(device)

    def initialize_token_loss(self):
        self.token_loss = th.zeros(self.steps, self.diff_config['model']['length'], device=device)
        self.token_count = th.zeros(self.steps, self.diff_config['model']['length'], device=device)

    def FullBlockMask(self, sequence_length, block_size, precursor_token=True):
        quadrant_1 = self.BlockMasks('offset_block_causal', sequence_length, block_size)
        if quadrant_1 == None:
            return None
        quadrant_2 = self.BlockMasks('block_diagonal', sequence_length, block_size)
        quadrant_3 = th.full_like(quadrant_1, 1e7)
        quadrant_4 = self.BlockMasks('block_causal', sequence_length, block_size)
        upper = th.cat([quadrant_2, quadrant_1], dim=1)
        lower = th.cat([quadrant_3, quadrant_4], dim=1)
        mask = th.cat([upper, lower], dim=0)
        if precursor_token:
            # Precursor can only see itself, nothing downstream
            mask = th.cat([th.zeros(mask.shape[0], 1, device=device), mask], dim=1)
            horizontal = th.cat([th.zeros(1), th.full((mask.shape[0],), 1e7)])[None].to(device)
            mask = th.cat([horizontal, mask], dim=0)
        return mask.to(device)

    def inptarg(self, batch):
        bs, sl = batch['intseq'].shape
        
        #input_tokens, output_tokens, new_mask = self.model.diff_obj._maybe_sub_sample(self, batch['intseq']) # Unnecessary, I think
        
        target = deepcopy(batch['intseq'])
        target = self.model.decoder.append_null_token(target)
        target = self.model.decoder.replace_with_eos_token(target, batch['peplen'])
        
        loss_mask = self.model.decoder.sequence_mask(target)
        
        return None, target, loss_mask

    def dropout_attributes(self, batch):
        batch_size = batch['mass'].shape[0]
        if self.model.decoder.diff_obj.use_guidance:
            dropout_mask = th.rand(batch_size) < self.diff_config['guidance']['p_uncond']
            if 'mass' in batch: batch['mass'][dropout_mask] = 0.
            if 'charge' in batch: batch['charge'][dropout_mask] = 0
            if 'mz' in batch: batch['mz'][dropout_mask] = 0.
            if 'ab' in batch: batch['ab'][dropout_mask] = 0.
        return batch

    def train_step(self, batch):
        block_decoding = True if self.model.decoder.block_size is not None else False
        batch = U.Dict2dev(batch, device)
        _, target, loss_mask = self.inptarg(batch)
        training_mask = self.FullBlockMask(target.shape[1], self.model.decoder.block_size, True)[None,None] if block_decoding else None
        
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()
        
        batch = self.dropout_attributes(batch)
        embedding = self.model.encoder_embedding(batch)
        
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding['emb'],
            'seqmask': training_mask,
            'doubled': True if block_decoding else False,
        }
        model_kwargs = self.dropout_attributes(model_kwargs)
        
        backbone = self.model.decoder
        if self.diff_config['custom_loss']:
            model_output, weights, masked_token_mask, timesteps = self.model.diff_obj._forward_pass_diffusion(backbone, target, model_kwargs, block_decoding)
            loss = F.cross_entropy(model_output.transpose(-1,-2), target, reduction='none')
            weights = (target!=self.model.decoder.NT).float() + 0.01*(target==self.model.decoder.NT).float()
            loss = (weights*loss)[masked_token_mask]
            other_losses = {}
        else:
            loss, other_losses = self.model.diff_obj._forward_pass_diffusion(backbone, target, model_kwargs, block_decoding)


        token_nll = loss.mean()
        losses = {'loss': token_nll.item()} | other_losses
        
        token_nll.backward()
        self.update_lr()
        self.opt.step()
        
        return losses
    
    def log_wandb(self, losses, grad_norm):
        loss = losses.pop('loss')
        wandb.log({
            "Total loss": loss,
            'Global step': self.global_step,
            "Global grad norm": grad_norm,
        } | losses)

    def on_train_epoch_end(self):
        if self.log:
            try:
                avg_loss = self.token_loss / (self.token_count+1e-5)
                avg_loss = avg_loss.cpu().detach().numpy()
                np.savetxt(os.path.join(self.svdir, "token_loss.tsv"), avg_loss, delimiter='\t')
                self.initialize_token_loss()
            except:
                pass

class DenovoD3PMObj(BaseDenovo):
    def __init__(self, config, svdir='./save/', rddir=None):
        super().__init__(
            config=config, 
            svdir=svdir,
            rddir=rddir,
        )
        self.training_loss_keys.extend(['loss'])
        self.eval_kwargs = {}

        from models.seq2seq import Seq2SeqD3PM
        
        diff_config = config['decoder_d3pm']
        self.diff_config = diff_config
        self.max_length = config['pep_length'][1]
        self.steps = diff_config['steps']
        #config['decoder_diff']['diffusion_config']['pad_tok_id'] = self.data.amod_dic['X']
        #config['decoder_diff']['diffusion_config']['sequence_len'] = self.config['pep_length'][1] + 1 # b/c of eos token
        
        # The diffusion models share the model_config, but with a la carte alterations
        config['decoder_diff']['model_config']['self_condition'] = diff_config['self_condition']
        
        self.model = Seq2SeqD3PM(
            encoder_config = config['encoder_dict'],
            decoder_config = config['decoder_diff']['model_config'],
            diff_config = diff_config,
            top_peaks = config['top_peaks'], 
            max_peptide_length = config['pep_length'][1], 
            token_dict = self.data.amod_dic,
            ensemble_config   = config['decoder_diff']['ensemble'],
            masses_path = config['loader']['masses_path'],
        )

        # Optimizer
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        self.restore_model()

        self.model.to(device)

    def inptarg(self, batch):
        bs, sl = batch['intseq'].shape
        
        #input_tokens, output_tokens, new_mask = self.model.diff_obj._maybe_sub_sample(self, batch['intseq']) # Unnecessary, I think
        
        target = deepcopy(batch['intseq'])
        target = self.model.decoder.append_null_token(target)
        target = self.model.decoder.replace_with_eos_token(target, batch['peplen'])
        
        loss_mask = self.model.decoder.sequence_mask(target)
        
        return None, target, loss_mask

    def train_step(self, batch):
        block_decoding = False #True if self.model.decoder.block_size is not None else False
        batch = U.Dict2dev(batch, device)
        _, target, loss_mask = self.inptarg(batch)
        training_mask = self.FullBlockMask(target.shape[1], self.model.decoder.block_size, True)[None,None] if block_decoding else None
        
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()
        
        embedding = self.model.encoder_embedding(batch)
        
        model_kwargs = {
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
            'kv_features': embedding['emb'],
            'seqmask': training_mask,
            'doubled': True if block_decoding else False,
        }
        
        loss, other_losses = self.model.diff_obj.forward(x=target, model_kwargs=model_kwargs)
        
        losses = {'loss': loss.item()} | other_losses
        
        loss.backward()
        self.update_lr()
        self.opt.step()
        
        return losses
    
    def log_wandb(self, losses, grad_norm):
        loss = losses.pop('loss')
        wandb.log({
            "Total loss": loss,
            'Global step': self.global_step,
            "Global grad norm": grad_norm,
        } | losses)
