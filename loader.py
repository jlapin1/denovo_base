from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re
from glob import glob
import sys
import pandas as pd

def map_fn(example, tokenizer, dic=None, top=100, max_seq=50):
    ab = th.tensor(example['intensity_array'])
    ab_sort = (-ab).argsort()[:top]
    ab = ab[ab_sort]
    ab /= ab.max()
    spectrum_length = len(ab)
    mz = th.tensor(example['mz_array'])[ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    mz_ = th.zeros(top)
    mz_[:len(mz_sort)] = mz[mz_sort]
    ab_ = th.zeros(top)
    ab_[:len(ab_sort)] = ab[mz_sort]
    example['mz_array'] = mz_
    example['intensity_array'] = ab_
    example['precursor_charge'] = th.tensor(example['precursor_charge'], dtype=th.int32)
    example['precursor_mass'] = th.tensor(example['precursor_mass'], dtype=th.float32)
    example['spectrum_length'] = th.tensor(len(example['mz_array']), dtype=th.int32)
    tokenized_sequence = tokenizer(example['modified_sequence'])
    peptide_length = len(tokenized_sequence)
    example['tokenized_sequence'] = th.tensor([dic[m] for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=th.int32)
    example['peptide_length'] = th.tensor(peptide_length, dtype=th.int32)
    example['spectrum_length'] = th.tensor(spectrum_length, dtype=th.int32)

    return example

def collate_fn(batch_list):
    species = [m['experiment_name'] for m in batch_list]
    speclen = th.stack([m['spectrum_length'] for m in batch_list])
    mz = th.stack([m['mz_array'][:speclen.max()] for m in batch_list])
    ab = th.stack([m['intensity_array'][:speclen.max()] for m in batch_list])
    charge = th.stack([m['precursor_charge'] for m in batch_list])
    mass = th.stack([m['precursor_mass'] for m in batch_list])
    peplen = th.stack([m['peptide_length'] for m in batch_list])
    intseq = th.stack([m['tokenized_sequence'][:peplen.max()] for m in batch_list])

    out = {
        'experiment_name': species,
        'mz': mz,
        'ab': ab,
        'charge': charge,
        'mass': mass,
        'length': speclen,
        'intseq': intseq,
        'peplen': peplen,
        #'spectrum_lengths': speclen[:,None],
    }

    return out

exceptions = {
    'C(+57.02)': 'C+57.021',
    'M(+15.99)': 'M+15.995',
    'N(+.98)': 'N+0.984',
    'Q(+.98)': 'Q+0.984',
}

class LoaderHF:
    def __init__(self, 
        train_dataset_path: str,
        train_name: str=None,
        val_dataset_path: str=None,
        val_name: str=None,
        dictionary_path: str=None,
        masses_path: str=None,
        tokenizer_path: str=None,
        test_split_method: str='full_val',
        top_pks: int=100,
        batch_size: int=100,
        num_workers: int=0,
        **kwargs
    ):
        if val_dataset_path is None:
            val_dataset_path = train_dataset_path
        if masses_path is None:
            masses_path = train_dataset_path
        tokenizer_path = train_dataset_path if tokenizer_path==None else tokenizer_path
        max_seq = kwargs['pep_length'][1] if 'pep_length' in kwargs.keys() else None

        # Scratch directory
        if 'scratch' in kwargs.keys():
            if kwargs['scratch']['use']:
                pth = kwargs['scratch']['path']
                if os.path.exists(pth):
                    # Change the dataset paths
                    dataset_path = {
                        key: pth + dataset_path[key].split("/")[-1]  
                        for key in dataset_path
                    }
                else:
                    print("Scratch directory not found. Using original paths.")

        # Dictionary
        if dictionary_path is not None:
            self.amod_dic = {
                line.split()[0]:m for m, line in enumerate(open(dictionary_path))
            }
            self.amod_dic['X'] = len(self.amod_dic)
            self.amod_dic_rev = {b:a for a,b in self.amod_dic.items()}

        # Dictionary masses
        # - RULES
        #   1. There is a file that matches the regex *masses.txt in the masses_path
        try:
            masses_path = glob(os.path.join(masses_path, "*masses.tsv"))[0]
            mass_frame = pd.read_csv(masses_path, delimiter="\t", header=None)
            self.massdic = {m:n for m,n in zip(mass_frame[0], mass_frame[1])}
        except:
            pass

        # Split sizes
        # - RULES
        #   1. There is a file that matches the regex *sizes.tsv in the train_dataset_path and val_dataset_path
        #   2. val_name will pick out 1 file's size from the val_dataset_path
        
        ss_train_path = os.path.join(train_dataset_path, "*sizes.tsv")
        ss_train_path = glob(ss_train_path)[0]
        if os.path.exists(ss_train_path):
            train_split_sizes = pd.read_csv(ss_train_path, sep="\t", header=None, names=["name", "count"], index_col="name")
            # if none, then read everything except for val_name
            # search train_split_sizes based on val_name to accomodate 9 species 
            #  cross validation where val and train are in the same directory
            if train_name == None:
                self.train_size = int(train_split_sizes.query(f"name.str.contains('{val_name}')==False")['count'].sum())
            # else affirmatively find files that contain train_name
            else:
                self.train_size = int(train_split_sizes.query(f"name.str.contains('{train_name}')")['count'].sum())
        else:
            # This should still work with tqdm progress bar
            self.train_size = float('inf')
        
        ss_val_path = os.path.join(val_dataset_path, "*sizes.tsv")
        ss_val_path = glob(ss_val_path)[0]
        if os.path.exists(ss_val_path):
            val_split_sizes = pd.read_csv(ss_val_path, sep='\t', header=None, names=['name', 'count'], index_col="name")
            self.val_size = int(val_split_sizes.query(f"name.str.contains('{val_name}')")['count'].sum())
        else:
            # This should still work with tqdm progress bar
            self.train_size = float('inf')
            self.val_size = float('inf')

        # Dataset
        # - RULES
        #   1. The *_directory_path will contain its data in a directory named "parquet/processed"
        #   2. val_name only has to be somewhere in the filename -> *val_name*
        # Read all files
        regex = "*" if train_name == None else f"*{train_name}*"
        train_dataset_path_ = os.path.join(train_dataset_path, "parquet/processed", regex)
        train_files = glob(train_dataset_path_)
        # Read only files matching *val_name*
        val_dataset_path_ = os.path.join(val_dataset_path, "parquet/processed", f"*{val_name}*")
        val_files = glob(val_dataset_path_)
        # If val files are in same directory as train files
        for val_file in val_files:
            if val_file in train_files: train_files.remove(val_file)
        print(f"<LOADCOMMENT> Found {len(train_files)} file(s) for training")
        print(f"<LOADCOMMENT> Found {len(val_files)} file(s) for validation")
        data_files = {'train': train_files, 'val': val_files,}
        
        dataset = load_dataset(
            'parquet',
            data_files={'train': data_files['train']},
            streaming=True
        )
        dataset_val = load_dataset(
            'parquet',
            data_files={'val': data_files['val']},
            streaming=True,
        )
        dataset['val'] = dataset_val['val']

        # Map to format outputs
        lambda_function = lambda example: map_fn(
            example,
            tokenizer=self.tokenizer,
            dic=self.amod_dic,
            top=top_pks, 
            max_seq=max_seq
        )
        if 'remove_columns' in kwargs:
            remove_train_columns = [column for column in kwargs['remove_columns'] if column in dataset['train'].features]
            remove_val_columns = [column for column in kwargs['remove_columns'] if column in dataset['val'].features]
        dataset['train'] = dataset['train'].map(
            lambda_function, 
            remove_columns=remove_train_columns,
        )
        dataset['val'] = dataset['val'].map(
            lambda_function,
            remove_columns=remove_val_columns,
        )

        # Create test from val
        if test_split_method == 'full_val':
            dataset['test'] = dataset['val']
        elif test_split_method == 'every_other':
            dataset['val'] = dataset['val'].filter(lambda example, idx: idx % every_n == 0, with_indices=True)
            dataset['test'] = dataset['test'].filter(lambda example, idx: idx % every_n == 1, with_indices=True)
        
        # Tokenizer
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        sys.path.append(tokenizer_path)
        from enumerate_tokens import partition_modified_sequence
        self.tokenizer = partition_modified_sequence

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (len(example['tokenized_sequence']) >= kwargs['pep_length'][0]) &
                (len(example['tokenized_sequence']) <= kwargs['pep_length'][1])
            )
        
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['precursor_charge'] >= kwargs['charge'][0]) &
                (example['precursor_charge'] <= kwargs['charge'][1])
            )

        # Filter val set for dispersed examples
        if 'val_steps' in kwargs.keys():
            if kwargs['val_steps'] is not None:
                every_n = self.val_size // batch_size // kwargs['val_steps'] - 1 # minus 1 to be safe (charge and length filter make dataset shorter)
                dataset['val'] = dataset['val'].filter(lambda example, idx: idx % every_n == 0, with_indices=True)
        
        # Shuffle the dataset
        if 'buffer_size' in kwargs.keys():
            dataset['train'] = dataset['train'].shuffle(buffer_size=kwargs['buffer_size'])
        else:
            dataset['train'] = dataset['train'].shuffle()
        
        self.dataset = dataset

        # Dataloaders
        num_workers = min(self.dataset['train'].n_shards, num_workers)
        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, num_workers),
            'val':   self.build_dataloader(dataset['val']  , batch_size, 0),
            'test':  self.build_dataloader(dataset['test'] , batch_size, 0),
        }

    def build_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

