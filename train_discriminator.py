"""
TODO
- Fix error of variables created on non-first call when training encoder
"""
import torch as th
import yaml
import path
from loader import LoaderDis
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
        "dataset_path": "/cmnfs/data/proteomics/foundational_model/9_species_V1",
        "dictionary_path": "/cmnfs/data/proteomics/foundational_model/9_species_V1/ns_dictionary.txt",
        'pep_length': [6,40],
    }
    loader = LoaderDis(**load_config)

    #########
    # Model #
    #########



if __name__ == '__main__':

    main()
