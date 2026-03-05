import os
import shutil
import math
from glob import glob
from collections import deque
from time import time
from copy import deepcopy

import torch as th
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
from tqdm import tqdm

from loader import LoaderHF
from loader_lance import LoaderLance
from models.diff_classifier import Classifier
from models.mdlm.train_backends import create_mdlm_train_wrapper
import metrics as met
import utils as U

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def set_runtime_device(new_device):
    global device
    device = new_device
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
            if val_steps == -1:
                self.val_steps = float('inf')
            else:
                self.val_steps = 1 if val_steps == None else val_steps # backwards compatiblity
        else:
            self.val_steps = 100
        self.reverse = config['loader']['reverse']
        use_lance_loader = self.config['loader'].get('use_lance', False)
        loader_cls = LoaderLance if use_lance_loader else LoaderHF
        self.data = loader_cls(
            top_pks=config['top_peaks'], 
            pep_length=None if (config['inference']&config['eval_only']) else config['pep_length'],
            batch_size=config['batch_size'],
            **self.config['loader']
        )
        
        self.training_loss_keys = []
        self.eval_stats = []
        self.eval_kwargs = {}
        self.distributed = U.dist_is_initialized()
        self.rank = U.get_rank()
        self.world_size = U.get_world_size()
        self.is_main = U.is_main_process()

    def get_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save_weights(self, fp='./model.wts'):
        if not self.is_main:
            return
        model = self.model.module if isinstance(self.model, DDP) else self.model
        th.save(model.state_dict(), fp)
    
    def save_last(self, override=False):
        ready = self.global_step - self.save_last_counter >= self.config['save_last_freq']
        should_save = bool(ready or override)
        if not should_save:
            return
        if self.distributed:
            # Ensure all ranks reached the same training step before rank-0 writes.
            U.dist_barrier()
        if self.is_main:
            self.save_weights(os.path.join(self.svdir, 'weights/model_last.wts'))
            U.save_optimizer_state(self.opt, os.path.join(self.svdir, 'weights/opt_last.wts'))
        self.save_last_counter = self.global_step
        if self.distributed:
            U.dist_barrier()
    
    def checkpoint(self, score):
        self.save_last(override=True)
        if self.distributed:
            U.dist_barrier()
        if self.is_main and score > self.high_score:
            self.high_score = score
            ext = f"step{self.global_step}_high_{self.high_score:.3f}"
            wtsdir = os.path.join(self.svdir, "weights")
            for file in glob(os.path.join(wtsdir, "*high*")): os.remove(file)
            self.save_weights(os.path.join(wtsdir, f"model_{ext}.wts"))
        if self.distributed:
            U.dist_barrier()

    def load_saved_weights(self, obj, weights_type='model', load_last=False, retain=False):
        regex = f'*{weights_type}*last*wts*' if load_last else f"*{weights_type}*wts*"
        if self.is_main:
            print(f"<DSCOMMENT> Searching for {weights_type} weights with regular expression {regex}")
        possible_weights_path = glob(os.path.join(self.rddir, "weights", regex))
        found_local = len(possible_weights_path) > 0
        if self.distributed:
            # Use SUM for broad NCCL compatibility; MIN/MAX on integer tensors can be backend-sensitive.
            found_tensor = th.tensor(float(found_local), device=device)
            found_sum = float(U.all_reduce_tensor(found_tensor, op=th.distributed.ReduceOp.SUM).item())
            if 0.0 < found_sum < float(self.world_size):
                raise RuntimeError(
                    f"Inconsistent checkpoint visibility across ranks for {weights_type}: "
                    f"rank{self.rank} found={found_local} path={self.rddir}"
                )
            found_local = found_sum > 0.0
        
        # Found something
        if found_local:
            
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
                    last_matches = [m for m in possible_weights_path if 'last' in m]
                    if len(last_matches) > 0:
                        weights_path = last_matches[0]
                        qualifier = '"last"'
                    else:
                        weights_path = sorted(possible_weights_path)[-1]
                        qualifier = "latest-sorted"
            
            if self.is_main:
                print(f"<DSCOMMENT> Loading {qualifier} previous {weights_type} weights: {weights_path}")
            target = obj.module if isinstance(obj, DDP) else obj
            # Load checkpoints on CPU first to avoid GPU-side load spikes on distributed startup.
            target.load_state_dict(th.load(weights_path, map_location='cpu', weights_only=False))

            if retain:
                try:
                    shutil.copyfile(weights_path, os.path.join(self.svdir, "weights", "save"))
                except:
                    pass
        
        # Found nothing
        else:
            if self.is_main:
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
        try:
            train_steps = len(self.data.dataloader['train'])
        except TypeError:
            train_steps = int(self.data.train_size // bs)
        max_train_batches = self.config.get('max_train_batches')
        if max_train_batches is not None:
            train_steps = min(train_steps, int(max_train_batches))
        pbar = tqdm(
            self.data.dataloader['train'], 
            total=train_steps, 
            smoothing=0.6, 
            disable=(not self.is_main),
        )

        epoch_start = time()
        step_end=epoch_start
        for step, batch in enumerate(pbar):      
            step_start = time()
            if max_train_batches is not None and step >= max_train_batches:
                break
            
            if self.config['log_wandb'] and self.is_main:
                wandb.log({"Learning rate": self.opt.param_groups[-1]['lr']})
            
            if self.config.get('debug_timing') and self.is_main:
                print(f"[debug] train_step {step} start")
            losses = self.train_step(batch)
            if self.config.get('debug_timing') and self.is_main:
                print(f"[debug] train_step {step} done in {time() - step_start:.2f}s")
            losses_reduced = U.reduce_dict(losses, average=True, device=device)
            self.global_step += 1
            
            if self.config['log_wandb']:
                loss_printout = 'Loss: %7f'%losses_reduced['loss']
                global_grad_norm = U.global_grad_norm(self.model)
                if self.distributed:
                    global_grad_norm = U.all_reduce_tensor(th.tensor(global_grad_norm, device=device)) / self.world_size
                    global_grad_norm = float(global_grad_norm.cpu().item())
                if self.is_main:
                    self.log_wandb(losses_reduced, global_grad_norm)
            else:
                if self.is_main:
                    for key in running_loss.keys(): running_loss[key].append(losses_reduced[key].detach().cpu())
                    rlm = {key: np.mean(running_loss[key]) for key in running_loss.keys()}
                    loss_printout = ", ".join(len(rlm)*['%s: %7f'])%tuple([m for n in rlm.items() for m in n])
            if self.is_main:
                pbar.set_description(f"Loss: {loss_printout}")
            
            if self.is_main:
                self.running_loss.append(losses_reduced['loss'].detach().cpu())
                if self.log and (self.global_step % svfreq == 0):
                    self.savetxt(self.running_loss)
                    self.running_loss = []
            if self.config['save_weights']:
                self.save_last()
            if self.eval_frequency is not None:
                local_eval_due = (time() - self.eval_time) > self.eval_frequency
                if self.distributed:
                    # Synchronize the eval trigger across all ranks to avoid train/eval divergence.
                    due_tensor = th.tensor(int(local_eval_due), device=device)
                    due_tensor = U.all_reduce_tensor(due_tensor, op=th.distributed.ReduceOp.MAX)
                    run_timed_eval = bool(due_tensor.item())
                else:
                    run_timed_eval = local_eval_due

                if run_timed_eval:
                    if self.distributed:
                        U.dist_barrier()
                    out, _ = self.evaluation(dset='val', max_batches=self.val_steps, kwargs=self.eval_kwargs)
                    self.eval_out = out
                    new_score = out[self.config['high_score']]
                    if self.config['save_weights']:
                        self.checkpoint(new_score)
                    self.eval_time = time()
                    if self.config['log_wandb'] and self.is_main:
                        wandb.log(out)
                    if self.distributed:
                        U.dist_barrier()
            
            step_end = time()
            
        if self.log and (len(self.running_loss) > 0) and self.is_main:
            self.savetxt(self.running_loss)
            self.running_loss = []
        
        if self.is_main:
            print("\rFinal running loss: %s, Final time elapsed: %.0f s"%(loss_printout, time()-epoch_start))
        
    def savetxt(self, train_loss=None, eval_stats=None):
        if not self.is_main:
            return
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
        model = self.get_model()
        intseq[eos_inds] = model.decoder.EOS

        return intseq

    def append_null_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        model = self.get_model()
        nulls = th.fill(th.empty(bs, dtype=th.int64), model.decoder.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def fill_null_after_first_eos_token(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        bs, sl = intseq.shape
        model = self.get_model()
        length = ((intseq == model.decoder.EOS)|(intseq == model.decoder.NT)).int().argmax(1)
        intseq[th.arange(bs), length] = model.decoder.EOS
        mask = (length > 0)[:,None]
        index_array = th.arange(sl)[None].repeat([bs, 1]).to(intseq.device)
        boolean_array = index_array > length[:, None]
        intseq[mask & boolean_array] = model.decoder.NT

        return intseq

    def to_list_of_strings(self, intseq):
        if len(intseq.shape) == 1:
            intseq = intseq[None]
        is_reverse = lambda x: x[::-1] if self.reverse else x
        model = self.get_model()
        return [
            is_reverse([
                model.decoder.rev_outdict[int(n)] 
                for n in m if n not in [model.decoder.NT, model.decoder.EOS]
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
        if self.distributed:
            # Avoid filename collisions; write one file per rank
            rank = self.rank
            if output_filename is not None:
                if output_filename.endswith(".parquet"):
                    output_filename = output_filename[:-8] + f".rank{rank}.parquet"
                else:
                    output_filename = f"{output_filename}.rank{rank}"

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
        model = self.get_model()
        model.eval()
        for i, batch in enumerate(pbar):
            if i >= max_batches:
                break

            #print("\rEvaluation step %d"%(i+1), end='')
            batchdev = U.Dict2dev(batch, device)
            if no_grad:
                with th.no_grad():
                    #seqint, target, loss_mask = self.inptarg(batchdev)
                    out_dict = model.predict_sequence(batchdev, **kwargs)
            else:
                #seqint, target, loss_mask = self.inptarg(batchdev)
                out_dict = model.predict_sequence(batchdev, **kwargs)
            
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
        if stream_write and schema_defined:
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
        if (save_df or stream_write) and self.is_main:
            dataframe = initial_dataframe()
            schema_defined = False
        else:
            dataframe = None

        # losses
        out = {'ce': th.tensor(0.0, device=device)}
        token_count = th.tensor(0.0, device=device)
        tots = {'sum':{}, 'total': {}}

        # Build a fresh eval dataloader each call to avoid iterator exhaustion
        # across epochs when using streaming datasets.
        base_loader = self.data.dataloader[dset]
        dataloader = self.data.build_dataloader(
            self.data.dataset[dset],
            base_loader.batch_size,
            0,
            base_loader.collate_fn,
        )

        # Progress bar
        try:
            val_steps = len(dataloader)
        except TypeError:
            val_steps = self.data.val_size // dataloader.batch_size
        val_steps = min(val_steps, max_batches)
        pbar = tqdm(dataloader, total=val_steps, leave=False, disable=(not self.is_main))
        pbar.set_description(f"Evaluation")
        model = self.get_model()
        model.eval()
        steps = 0
        for i, batch in enumerate(pbar):
            if i >= max_batches:
                break
            steps += 1

            #print("\rEvaluation step %d"%(i+1), end='')
            batchdev = U.Dict2dev(batch, device)
            if no_grad:
                with th.no_grad():
                    seqint, target, loss_mask = self.inptarg(batchdev)
                    out_dict = model.predict_sequence(batchdev, **kwargs)
            else:
                seqint, target, loss_mask = self.inptarg(batchdev)
                out_dict = model.predict_sequence(batchdev, **kwargs)
            prediction = out_dict.pop('prediction')
            probs = out_dict.pop('logits')
            
            # Do some resizing/reshaping
            prediction = prediction[..., :target.shape[1]] # loaded shapes can change based on batch
            probs = probs[:, :target.shape[1]]
            predicted_probs = probs.softmax(-1).gather(-1, prediction[...,None].type(th.int64)).squeeze(-1)
            
            # Cross entropy
            pred = probs.transpose(-1,-2)
            ce_sum = F.cross_entropy(pred, target, reduction='none')[loss_mask].sum()
            out['ce'] += ce_sum
            token_count += loss_mask.sum()
            
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

            if (save_df or stream_write) and self.is_main:
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
				
                if stream_write and ((i+1) % batches_btw_write == 0) and self.is_main:
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
        
        steps_tensor = th.tensor(float(steps), device=out['ce'].device)
        if self.distributed:
            steps_tensor = U.all_reduce_tensor(steps_tensor, op=th.distributed.ReduceOp.SUM)
            out['ce'] = U.all_reduce_tensor(out['ce'].detach(), op=th.distributed.ReduceOp.SUM)
            token_count = U.all_reduce_tensor(token_count.detach(), op=th.distributed.ReduceOp.SUM)
            for metric in tots['sum'].keys():
                sum_t = th.tensor(float(tots['sum'][metric]), device=out['ce'].device)
                tot_t = th.tensor(float(tots['total'][metric]), device=out['ce'].device)
                sum_t = U.all_reduce_tensor(sum_t, op=th.distributed.ReduceOp.SUM)
                tot_t = U.all_reduce_tensor(tot_t, op=th.distributed.ReduceOp.SUM)
                tots['sum'][metric] = float(sum_t.cpu().item())
                tots['total'][metric] = float(tot_t.cpu().item())
        steps_val = int(steps_tensor.cpu().item())
        if steps_val == 0:
            raise RuntimeError("No evaluation batches were processed; check dataset, sharding, and val_steps.")
        token_count_val = float(token_count.cpu().item())
        if token_count_val == 0:
            out['ce'] = float('nan')
            for metric in tots['sum'].keys():
                out[metric] = float('nan')
        else:
            out['ce'] = float((out['ce'] / token_count_val).cpu().detach().numpy())
            for metric in tots['sum'].keys():
                out[metric] = tots['sum'][metric] /  tots['total'][metric]
        
        self.on_eval_end()
        
        if stream_write and self.is_main and (len(dataframe['name']) > 0):
            table = pa.Table.from_pandas(pd.DataFrame(dataframe), preserve_index=False)
            writer.write_table(table)
            writer.close()
            return out, None
        elif save_df and self.is_main:   
            return out, pd.DataFrame(dataframe)
        else:
            return out, None

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        highline = None
        for i in range(self.config['epochs']):
            
            # Train
            if hasattr(self.data, "set_epoch"):
                self.data.set_epoch(i)
            elif hasattr(self.data.dataset['train'], "set_epoch"):
                self.data.dataset['train'].set_epoch(i)
            self.train_epoch()
            self.on_train_epoch_end()
            
            # Eval
            out = None
            new_score = None
            if self.eval_frequency is None:
                out, _ = self.evaluation(dset=eval_dset, max_batches=self.val_steps, kwargs=self.eval_kwargs)
                new_score = out[self.config["high_score"]]
                if self.config['save_weights']: self.checkpoint(new_score)
            
                # Logging
                if self.config['log_wandb'] and self.is_main:
                    out['epoch'] = i+1
                    wandb.log(out)
                    out.pop('epoch')
            else:
                # When time-triggered eval is enabled, do not run an extra epoch-end eval.
                out = self.eval_out if hasattr(self, 'eval_out') else None
                if out is not None:
                    new_score = out[self.config["high_score"]]

            if out is None:
                line = "ValEpoch %d: skipped (waiting for time-triggered eval)" % i
                line += " (%.1f s)"%(time()-start_time)
                lines.append(line)
                if self.is_main:
                    print("\r"+line)
                continue
            
            specifier = " ".join(len(out)*['%s'])
            write_out = specifier%tuple([f"{m}={n:.3}" for m,n, in out.items()])
            line = "ValEpoch %d: %s"%(i, write_out)
            
            if new_score > self.high_score:
                highline = line
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            if self.is_main:
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
            if self.log and self.is_main:
                self.savetxt(train_loss=None, eval_stats=np.array(self.eval_stats))
            
        if highline is None:
            highline = lines[-1] if len(lines) > 0 else ""
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
            use_precomputed_encoder=config['loader'].get('use_precomputed_encoder', False),
            precomputed_encoder_dim=config['loader'].get('precomputed_encoder_dim'),
            precomputed_kv_indim=config['loader'].get('precomputed_kv_indim'),
        )
        
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")

        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)

        self.predict_sequence = self.model.decoder.predict_sequence
        
        # loading previous weights
        if config['prev_wts'] is not None:
            retain = False if config['load_last'] else True
            self.load_saved_weights(self.model, "model", config['load_last'], retain=retain)
            if config['load_last']:
                self.load_saved_weights(self.opt, "opt", config['load_last'])     
                U.optimizer_to(self.opt, device)
        
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
        
        return {'loss': loss}

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
            use_precomputed_encoder=config['loader'].get('use_precomputed_encoder', False),
            precomputed_encoder_dim=config['loader'].get('precomputed_encoder_dim'),
            precomputed_kv_indim=config['loader'].get('precomputed_kv_indim'),
        )

        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)
        
        # loading previous weights
        if config['prev_wts'] is not None:
            retain = False if config['load_last'] else True
            self.load_saved_weights(self.model, "model", config['load_last'], retain=retain)
            if config['load_last']:
                self.load_saved_weights(self.opt, "opt", config['load_last'])
                U.optimizer_to(self.opt, device)
        
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
        if 'custom_loss' not in diff_config:
            diff_config['custom_loss'] = True
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
            use_precomputed_encoder=config['loader'].get('use_precomputed_encoder', False),
            precomputed_encoder_dim=config['loader'].get('precomputed_encoder_dim'),
            precomputed_kv_indim=config['loader'].get('precomputed_kv_indim'),
        )
        self.initialize_token_loss()
        
        # Moving average of weights
        import models.mdlm.ema as ema
        import itertools
        if diff_config['training']['ema'] > 0:
            ema_param_groups = [self.model.decoder.parameters(), self.model.diff_obj.noise.parameters()]
            if getattr(self.model, "encoder", None) is not None:
                ema_param_groups.insert(0, self.model.encoder.parameters())
            if getattr(self.model, "precomputed_proj", None) is not None:
                ema_param_groups.append(self.model.precomputed_proj.parameters())
            self.ema = ema.ExponentialMovingAverage(
                itertools.chain(*ema_param_groups),
                decay=diff_config['training']['ema']
            )
        
        # Optimizer
        print(f"<DSCOMMENT> Total model parameters: {self.model.total_params():,}")
        self.opt = th.optim.Adam(self.model.parameters(), self.starting_lr)
        
        # loading previous weights
        if config['prev_wts'] is not None:
            retain = False if config['load_last'] else True
            self.load_saved_weights(self.model, "model", config['load_last'], retain=retain)
            self.load_saved_weights(self.opt, "opt", config['load_last'])
            U.optimizer_to(self.opt, device)
        
        self.model.to(device)
        # Ensure diffusion object uses correct device
        self.model.diff_obj.device = device
        # DDP training wrapper (keeps inference model unwrapped)
        self.train_backend_name = self.diff_config.get('backend', 'legacy_mdlm_mask')
        self.train_wrapper = create_mdlm_train_wrapper(
            self.model,
            backend=self.train_backend_name,
        ).to(device)
        self.ddp_train_wrapper = None
        if self.distributed:
            if device.type == "cuda":
                self.ddp_train_wrapper = DDP(
                    self.train_wrapper,
                    device_ids=[device.index],
                    output_device=device.index,
                )
            else:
                self.ddp_train_wrapper = DDP(self.train_wrapper)

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
        model = self.get_model()
        target = model.decoder.append_null_token(target)
        target = model.decoder.replace_with_eos_token(target, batch['peplen'])
        
        loss_mask = model.decoder.sequence_mask(target)
        
        return None, target, loss_mask

    def train_step(self, batch):
        model = self.get_model()
        block_decoding = True if model.decoder.block_size is not None else False
        batch = U.Dict2dev(batch, device)
        _, target, loss_mask = self.inptarg(batch)
        training_mask = self.FullBlockMask(target.shape[1], model.decoder.block_size, True)[None,None] if block_decoding else None
        
        model.to(device)
        if self.ddp_train_wrapper is not None:
            self.ddp_train_wrapper.train()
        else:
            model.train()
        model.zero_grad()
        
        if self.ddp_train_wrapper is not None:
            loss = self.ddp_train_wrapper(batch, target, training_mask, block_decoding, self.diff_config['custom_loss'])
        else:
            loss = self.train_wrapper(batch, target, training_mask, block_decoding, self.diff_config['custom_loss'])
        token_nll = loss.mean()
        losses = {'loss': token_nll}
        
        token_nll.backward()
        self.update_lr()
        self.opt.step()
        
        return losses
    
    def log_wandb(self, losses, grad_norm):
        wandb.log({
            "Total loss": losses['loss'],
            'Global step': self.global_step,
            "Global grad norm": grad_norm,
        })

    def on_train_epoch_end(self):
        if self.log:
            try:
                avg_loss = self.token_loss / (self.token_count+1e-5)
                avg_loss = avg_loss.cpu().detach().numpy()
                if self.is_main:
                    np.savetxt(os.path.join(self.svdir, "token_loss.tsv"), avg_loss, delimiter='\t')
                self.initialize_token_loss()
            except:
                pass
