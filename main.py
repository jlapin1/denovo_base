"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loader import LoaderHF
from loader_lance import LoaderLance
import numpy as np
from models.encoder import Encoder
from models.diff_classifier import Classifier
from models.diff_decoder import DenovoDiffusionDecoder
from models.decoder import DenovoDecoder
import os
import sys
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
from torch.nn.parallel import DistributedDataParallel as DDP
from runners.denovo_objects import (
    DenovoArDSObj,
    DenovoDiffusionObj,
    DenovoMDLMObj,
    set_runtime_device,
)
from runners.d3pm_objects import DenovoD3PMObj
from runners.insertdelete_objects import DenovoInsertDeleteObj
nn = th.nn
F = nn.functional
choice = np.random.choice
device = th.device("cuda" if th.cuda.is_available() else "cpu")



if __name__ == '__main__':
    
    ##############
    # Read yamls #
    ##############

    #######################
    # Configuration files #
    #######################

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./yaml/config.yaml"
    overrides = U._parse_overrides(sys.argv[2:]) if len(sys.argv) > 2 else []

    # Read yamls
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    created_paths = U._apply_overrides(config, overrides) if overrides else []
    if 'pre_train_eval' not in config:
        config['pre_train_eval'] = False
    # Overrides over a loaded previous experiment
    config_ = deepcopy(config)
    # Eval config will not be overwritten
    with open("./yaml/eval.yaml") as stream:
        evconfig = yaml.safe_load(stream)

    #############################################
    # Distributed initialization and device set #
    #############################################
    dist_info = U.init_distributed()
    device = U.get_device()
    set_runtime_device(device)
    is_main = U.is_main_process()
    if dist_info["distributed"] and th.distributed.is_initialized():
        U.dist_barrier()
    if overrides and is_main and created_paths:
        print(f"<DSCOMMENT> CLI overrides created new keys: {sorted(set(created_paths))}")

    ########################################################
    # Create experiment directory in save/downstream_only/ #
    ########################################################

    # Continuing previous downstream run
    timestamp = U.timestamp(include_microseconds=True)
    checkpoint_root = config.get('checkpoint_root', 'save')
    if checkpoint_root in [None, '']:
        checkpoint_root = 'save'
    if config['prev_wts'] is not None:
        rddir = os.path.join(config['prev_wts'])
        if config['new_exp']:
            svdir = U.unique_experiment_dir(checkpoint_root, timestamp) if is_main else None
            if dist_info["distributed"] and th.distributed.is_initialized():
                svdir_obj = [svdir]
                th.distributed.broadcast_object_list(svdir_obj, src=0)
                svdir = svdir_obj[0]
            timestamp = os.path.basename(svdir)
            if not config['eval_only'] and is_main:
                U.create_experiment(svdir, svwts=config['save_weights'])
                print("<DSCOMMENT> Experiment is writing to directory %s"%svdir)
        else:
            svdir = os.path.join(config['prev_wts'])
            timestamp = config['prev_wts']
        with open(os.path.join(config['prev_wts'], "yaml", "config.yaml")) as stream:
            config = yaml.safe_load(stream)
        # Replace previous settings with new ones
        for key in [
            'epochs', 'prev_wts', 'load_last', 'lr_schedule',
            'lr_warmup_start', 'lr_warmup_end', 'lr_warmup_steps',
            'lr_flat_steps', 'lr_floor', 'lr_decay_steps',
            'loader', 'log_wandb', 'eval_only', 'batch_size',
            'top_peaks', 'classifier_config', 'new_exp', 'inference',
        ]:
            if key == 'loader':
                # These must be consistent with embedding layer in decoder
                config_[key]['synonyms'] = config[key]['synonyms']
                config_[key]['dictionary_path'] = config[key]['dictionary_path']
                config_[key]['reverse'] = config[key]['reverse']
            config[key] = config_[key]
        if overrides:
            U._apply_overrides(config, overrides)
            
    # Create new experiment
    elif config['save_weights'] and not config['eval_only']:
        rddir = None
        svdir = U.unique_experiment_dir(checkpoint_root, timestamp) if is_main else None
        if dist_info["distributed"] and th.distributed.is_initialized():
            svdir_obj = [svdir]
            th.distributed.broadcast_object_list(svdir_obj, src=0)
            svdir = svdir_obj[0]
        timestamp = os.path.basename(svdir)
        if is_main:
            U.create_experiment(svdir, svwts=config['save_weights'])
            print("<DSCOMMENT> Experiment is writing to directory %s"%svdir)
    else:
        rddir = None
        svdir = './'

    if dist_info["distributed"] and th.distributed.is_initialized():
        U.dist_barrier()

    # Eval only. Must set before loader is created.
    if config['eval_only']:
        config['loader']['val_dataset_path'] = evconfig['eval_only']['eval_dataset_path']
        config['loader']['val_name'] = evconfig['eval_only']['eval_name']
        cc = evconfig['eval_only']['loader_custom_columns']
        config['loader']['custom_columns'] = [] if cc == None else cc
        config['loader']['val_steps'] = evconfig['eval_only']['val_steps']
        config['loader']['disperse'] = evconfig['eval_only']['disperse']
    
    #####################
    # Downstream object #
    #####################

    print("<DSCOMMENT> Denovo sequencing")
    if 'insertdelete' in config['decoder_name']:
        print("<DSCOMMENT> Using insert/delete diffusion decoder")
        D = DenovoInsertDeleteObj(config, svdir=svdir, rddir=rddir)
    elif 'd3pm' in config['decoder_name']:
        print("<DSCOMMENT> Using D3PM decoder")
        D = DenovoD3PMObj(config, svdir=svdir, rddir=rddir)
    elif 'diff' in config['decoder_name']:
        print("<DSCOMMENT> Using diffusion decoder")
        D = DenovoDiffusionObj(config, svdir=svdir, rddir=rddir)
    elif 'mdlm' in config['decoder_name']:
        print("<DSCOMMENT> Using masked diffusion language decoder")
        D = DenovoMDLMObj(config, svdir=svdir, rddir=rddir)
    else:
        print("<DSCOMMENT> Using autoregressive decoder")
        D = DenovoArDSObj(config, svdir=svdir, rddir=rddir)

    # WandB
    if config['log_wandb'] and (config['eval_only'] == False) and is_main:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
			config={
				'master': config,
                'save_directory': timestamp,
                'model_parameters': D.model.total_params(),
			},
		)   
    
    ##################################
    # Run training and/or evaluation #
    ##################################

    try:
        if config['eval_only']:
            evc = evconfig['eval_only']
            
            # Apply settings that are independent of training
            max_batches = int(eval(str(evc['val_steps'] if evc['val_steps'] is not None else 9e10)))
            if 'max_batches' in evc.keys(): max_batches = evc['max_batches'] # override val steps
            if config['decoder_name'] in ['diff', 'mdlm', 'd3pm', 'insertdelete']:
                if evc['clamp_denoised'] is not None:
                    D.model.decoder.clamp_denoised = evc['clamp_denoised']
                if evc['n'] is not None:
                    D.model.ens_size = evc['n']
                    D.eval_kwargs['n'] = D.model.ens_size
            
            # Turn gradients off for de novo model
            for parm in D.model.parameters(): parm.requires_grad=False
            
            # Classifier guidance
            no_grad = False if hasattr(D, 'classifier') else True

            # Run evaluation
            evalkwargs = dict(evc['eval_kwargs']) if evc['eval_kwargs'] is not None else {}
            if config['inference']:
                D.inference(
                    output_filename=evc['outpath'],
                    dset=evc['set'],
                    max_batches=max_batches,
                    stream_write=evc['stream'],
                    no_grad=no_grad,
                    kwargs=D.eval_kwargs|evalkwargs,
                    save_keys=evconfig['inference_save_keys'],
                )
            else:
                out, df = D.evaluation(
                    dset=evc['set'], 
                    max_batches=max_batches, 
                    save_df=evc['save'], 
                    stream_write=evc['stream'],
                    no_grad=no_grad, 
                    kwargs=D.eval_kwargs|evalkwargs,
                )
            
                # Saving results
                if evc['save']:
                    eval_out_path = evc['outpath'] if evc['outpath'] is not None else os.path.join(svdir, "output.parquet")
                    if evc['stream']:
                        os.system(f"mv ./hold.parquet {eval_out_path}")
                    else:
                        df.to_parquet(eval_out_path)
                print("\n", out)
        else:
            if config.get('pre_train_eval', False):
                print("Test validation", end='')
                out, _ = D.evaluation(dset='val', max_batches=D.val_steps, kwargs=D.eval_kwargs)
                assert D.config['high_score'] in out.keys()
                print("\rTest validation passed")
            print(D.TrainEval()[-1])
    finally:
        if U.dist_is_initialized():
            try:
                U.dist_barrier()
            except Exception:
                pass
            try:
                th.distributed.destroy_process_group()
            except Exception:
                pass
