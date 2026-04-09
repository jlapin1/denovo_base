"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loader import LoaderCls, LoaderRegr
from models.diff_classifier import Classifier, Regressor4MDLM
import numpy as np
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

def main():
    
    ##############
    # Dataloader #
    ##############

    load_config = {
        "dataset_path": "/cmnfs/data/proteomics/foundational_model/MassiveKB",
        "dictionary_path": "/cmnfs/data/proteomics/foundational_model/MassiveKB/dictionary.tsv",
        'pep_length': [6,40],
        'synonyms': [['I','L']],
    }
    loader = LoaderRegr(**load_config)
    batch  = next(iter(loader.dataloader['test']))
    print(batch)

    #########
    # Model #
    #########
    
    diff_dir = "save/2026-04-03_16-08-34"
    model = Regressor4MDLM(
        diff_dir,
        loader.amod_dic,
        null_token = loader.amod_dic['X'],
    )
    model.data_mean = th.nn.Parameter(th.tensor(loader.mean), requires_grad=False)
    model.data_std = th.nn.Parameter(th.tensor(loader.std), requires_grad=False)
    model.to(device)
    print(f"<MAINCOMMENT> Total classifier parameters: {model.total_params():,}")

    opt = th.optim.Adam(model.parameters(), 1e-5)
    
    #weights_path = "/cmnfs/proj/diffusion/experiments/2025-08-01_00-24-55/enzyme_classifier/weights/model_epoch=14_ce=1.2132122764480224.wts"
    #classifier.load_state_dict(th.load(weights_path, map_location=device, weights_only=False))

    ##############
    # Evaluation #
    ##############

    def evaluation():
        model.eval()

        sums = {'total_inst': 0, 'mse': 0, 'mae_real': 0}

        pbar = tqdm(loader.dataloader['test'], leave=False)
        pbar.set_description("Evaluation")
        for step, batch in enumerate(pbar):
            if step == 1000:
                break
            batchdev = U.Dict2dev(batch, device)
            bs, sl = batchdev['intseq'].shape
            sums[f'total_inst'] += bs
            
            xt, model_kwargs = model.sample_xt_from_x0(batchdev['intseq'])
            with th.no_grad():
                out = model(xt)
            sums['mse'] += (out['out']-batchdev['labels']).square().sum().item()
            real_mass = out['out']*model.data_std + model.data_mean
            sums['mae_real'] += (real_mass - batchdev['real_mass']).abs().sum().item()
            
        
        # Averages
        out = {}
        total = sums.pop('total_inst')
        for key in sums.keys():
            out[key] = sums[key] / total

        return out

    ##############
    # Train step #
    ##############
    
    def train_step(batch):
        opt.zero_grad()
        model.train()
        batchdev = U.Dict2dev(batch, device)
        
        xt, model_kwargs = model.sample_xt_from_x0(batchdev['intseq'])
        out = model(xt)
        loss = (out['out'] - batchdev['labels'].type(th.float32)).square().mean()

        loss.backward()
        opt.step()

        return loss

    ############
    # Training #
    ############

    def train(epochs=1):
        timestamp = U.timestamp()
        save_directory = os.path.join('save', timestamp)
        weights_directory = os.path.join(save_directory, "weights")
        U.create_experiment(save_directory, svwts=True)

        best_score = 1e10
        out = evaluation()
        for epoch in range(epochs):
            loader.dataset['train'].set_epoch(epoch)
            # Progress bar
            loader.dataset['train'] = loader.dataset['train'].shuffle()
            pbar = tqdm(loader.dataloader['train'], smoothing=0.1)
            for step, batch in enumerate(pbar):
                loss = train_step(batch)

                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.3f}")
            out = evaluation()
            print(out)
            if out['mse'] < best_score:
                best_score = out['mse']
                files = glob(os.path.join(weights_directory, "*.wts"))
                for file in files: os.remove(file)
                ckpt_name = f"model_epoch={epoch}_ce={best_score}.wts"
                save_weights(model, os.path.join(weights_directory, ckpt_name))
    #print(evaluation())
    train(100)

def save_weights(model, fp='./model.wts'):
    th.save(model.state_dict(), fp)

if __name__ == '__main__':
    main()
