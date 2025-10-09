"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loader import LoaderCls
from models.diff_classifier import Classifier
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
        "dataset_path": "/cmnfs/data/proteomics/foundational_model/InstaNovo",
        "dictionary_path": "/cmnfs/data/proteomics/foundational_model/kingdoms/dictionary.tsv",
        'pep_length': [6,40],
    }
    loader = LoaderCls(**load_config)

    batch  = next(iter(loader.dataloader['test']))
    print(batch)

    #########
    # Model #
    #########
    
    diff_dir = "/cmnfs/proj/diffusion/experiments/2025-08-01_00-24-55"
    classifier = Classifier(
        diff_dir, 
        num_input_tokens   = len(loader.amod_dic) + 1,
        num_output_classes = len(loader.label_dict),
        null_token         = loader.amod_dic['X'],
    )
    classifier.to(device)
    print(f"<MAINCOMMENT> Total classifier parameters: {classifier.total_params():,}")

    opt = th.optim.Adam(classifier.parameters(), 1e-5)

    ##############
    # Evaluation #
    ##############

    def evaluation():
        classifier.eval()

        timesteps = classifier.diff_obj.num_timesteps
        T = [0, int(timesteps//3), int(timesteps//1.5)]
        
        sums = {'total_inst': 0}
        for t in T:
            sums[f'ce_{t}'] = 0
            sums[f'correct_{t}'] = 0

        pbar = tqdm(loader.dataloader['test'], leave=False)
        for step, batch in enumerate(pbar):
            batchdev = U.Dict2dev(batch, device)
            bs, sl = batchdev['intseq'].shape
            sums[f'total_inst'] += bs
            for t in T:
                
                ts = th.full((bs,), int(t), dtype=th.int32, device=device)
                latents = classifier.get_noisy_x(batchdev['intseq'], ts)
                
                with th.no_grad():
                    out = classifier(latents, ts)
                
                #cross_entropy_loss = nn.functional.cross_entropy(out, batchdev['labels'].type(th.int64), reduction='none')
                cross_entropy_loss = nn.functional.binary_cross_entropy_with_logits(out, batchdev['labels'].type(th.float32), reduction='none')
                sums[f'ce_{t}'] += cross_entropy_loss.sum()

               # sums[f'correct_{t}'] += (out.argmax(-1) == batchdev['labels']).sum()

        
        # Averages
        out = {'ce_all': 0, 'accuracy_all': 0}
        for t in T:
            out[f'ce_{t}'] = float(sums[f'ce_{t}'])    / sums[f'total_inst']
            #out[f'accuracy_{t}'] = int(sums[f'correct_{t}']) / sums[f'total_inst']
            out['ce_all'] += out[f'ce_{t}']
            #out['accuracy_all'] += out[f'accuracy_{t}']
        out['ce_all'] /= len(T)
        out['accuracy_all'] /= len(T)

        return out

    ##############
    # Train step #
    ##############
    
    def train_step(batch):
        bs, sl = batch['intseq'].shape
        opt.zero_grad()
        batchdev = U.Dict2dev(batch, device)
        ts = th.empty(bs, device=device).uniform_(0, classifier.diff_obj.num_timesteps).type(th.int32)
        latents = classifier.get_noisy_x(batchdev['intseq'], ts)
        
        classifier.train()
        out = classifier(latents, ts)
        #loss = nn.functional.cross_entropy(out, batchdev['labels'].type(th.int64))
        loss = nn.functional.binary_cross_entropy_with_logits(out, batchdev['labels'].type(th.float32))

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
            # Progress bar
            loader.dataset['train'] = loader.dataset['train'].shuffle()
            pbar = tqdm(loader.dataloader['train'], smoothing=0.1)
            for step, batch in enumerate(pbar):
                loss = train_step(batch)

                pbar.set_description(f"Epoch {epoch}, Loss: {loss:.3f}")
            out = evaluation()
            print(out)
            if out['ce_all'] < best_score:
                best_score = out['ce_all']
                files = glob(os.path.join(weights_directory, "*.wts"))
                for file in files: os.remove(file)
                ckpt_name = f"model_epoch={epoch}_ce={best_score}.wts"
                save_weights(classifier, os.path.join(weights_directory, ckpt_name))

    train(100)

def save_weights(model, fp='./model.wts'):
    th.save(model.state_dict(), fp)

if __name__ == '__main__':
    main()
