from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re
from glob import glob
import sys
import pandas as pd
import numpy as np
join = os.path.join

def map_fn(example, tokenizer, dic=None, top=100, max_seq=50):
    ab = example['intensity_array']
    ab_sort = (-ab).argsort()[:top]
    ab = ab[ab_sort]
    ab /= ab.max()
    spectrum_length = len(ab)
    mz = example['mz_array'][ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    mz_ = np.zeros(top)
    mz_[:len(mz_sort)] = mz[mz_sort]
    ab_ = np.zeros(top)
    ab_[:len(ab_sort)] = ab[mz_sort]
    example['mz_array'] = mz_
    example['intensity_array'] = ab_
    example['precursor_charge'] = example['precursor_charge']
    example['precursor_mass'] = example['precursor_mass']
    example['spectrum_length'] = len(example['mz_array'])
    tokenized_sequence = tokenizer(example['modified_sequence'])
    peptide_length = len(tokenized_sequence)
    example['tokenized_sequence'] = np.array([dic.get(m, dic['X']) for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=np.int32)
    example['peptide_length'] = peptide_length
    example['spectrum_length'] = spectrum_length
    if 'name' in example: example['experiment_name'] = example['name'] # compat

    return example

def collate_fn(batch_list):
    species = [m['experiment_name'] for m in batch_list]
    speclen = np.stack([m['spectrum_length'] for m in batch_list])
    mz = np.stack([m['mz_array'][:speclen.max()] for m in batch_list])
    ab = np.stack([m['intensity_array'][:speclen.max()] for m in batch_list])
    charge = np.stack([m['precursor_charge'] for m in batch_list])
    mass = np.stack([m['precursor_mass'] for m in batch_list])
    peplen = np.stack([m['peptide_length'] for m in batch_list])
    intseq = np.stack([m['tokenized_sequence'][:peplen.max()] for m in batch_list])
    chimeric = np.stack([m['chimeric'] for m in batch_list])
    hyperscore = np.stack([m['Hyperscore'] for m in batch_list])

    out = {
        'experiment_name': species,
        'mz': th.tensor(mz, dtype=th.float32),
        'ab': th.tensor(ab, dtype=th.float32),
        'charge': th.tensor(charge, dtype=th.int32),
        'mass': th.tensor(mass, dtype=th.float32),
        'length': th.tensor(speclen, dtype=th.int32),
        'intseq': th.tensor(intseq, dtype=th.int32),
        'peplen': th.tensor(peplen, dtype=th.int32),
        'chimeric': chimeric,
        'hyperscore': hyperscore,
    }

    return out

class LoaderObj:
    def build_dataloader(self, dataset, batch_size, num_workers, collate_fn, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
    
    def create_sequence_dictionary(self, dictionary_path):
        amod_dic = {
            line.split()[0]:m for m, line in enumerate(open(dictionary_path).read().strip().split('\n'))
        }
        amod_dic['X'] = len(amod_dic)

        return amod_dic

    def reverse_dictionary(self, amod_dic):
        return {b:a for a,b in amod_dic.items()}

    def synonym(self, token1, token2, amod_dic):
        low = np.minimum(amod_dic[token1], amod_dic[token2])
        high = np.maximum(amod_dic[token1], amod_dic[token2])
        amod_dic[token1] = amod_dic[token2] = low
        amod_dic = {key:value if value < high else value-1 for key, value in amod_dic.items()}
        return amod_dic
    
    def create_label_dictionary(self, dataset_path):
        filepath = join(dataset_path, "parquet/labeled_sequences/species_list.txt")
        List = open(filepath).read().split("\n")
        #tsv = pd.read_csv(filepath, sep='\t', header=None, names=['species_name', 'count'])
        label_dict = {j:i for i,j  in enumerate(List)}
        label_dictr = {j:i for i,j in label_dict.items()}

        return label_dict, label_dictr
    
    def create_tokenizer(self, tokenizer_path):
        # Tokenizer
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        sys.path.insert(0, tokenizer_path)
        from enumerate_tokens import partition_modified_sequence
        tokenizer = partition_modified_sequence

        return tokenizer

    def load_token_masses(self, masses_path, regex='*masses.tsv'):
        try:
            masses_path = glob(join(masses_path, regex))[0]
            mass_frame = pd.read_csv(masses_path, delimiter="\t", header=None)
            massdic = {m:n for m,n in zip(mass_frame[0], mass_frame[1])}
        except:
            massdic = None

        return massdic

    def find_set_size_for_tqdm(self, dataset_path, include_name=None, exclude_name=None, regex='*sizes.tsv'):
        ss_path = join(dataset_path, regex)
        ss_path = glob(ss_path)[0]
        if os.path.exists(ss_path):
            split_sizes = pd.read_csv(ss_path, sep="\t", header=None, names=["name", "count"], index_col="name")
            
            # if None, then read everything except for val_name
            # search split_sizes based on val_name to accomodate 9 species 
            #  cross validation where val and train are in the same directory
            if include_name == None:
                size = int(split_sizes.query(f"name.str.contains('{exclude_name}')==False")['count'].sum())
            # else affirmatively find files that contain train_name
            else:
                size = int(split_sizes.query(f"name.str.contains('{include_name}')")['count'].sum())
        else:
            # This should still work with tqdm progress bar
            size = float('inf')

        return size

    def _load_dataset(self, dataset_path, include_name=None, exclude_name=None, ext=None):
        regex = "*" if include_name == None else f"*{include_name}*"
        if ext is not None:
            regex += ext
        dataset_path_ = join(dataset_path, regex)
        include_files = glob(dataset_path_)
        
        # If the train files cannot be specified without including the val file
        # --> affirmatively exclude it
        if exclude_name is not None:
            exclude_files = glob(join(dataset_path, f"*{exclude_name}*"))
            for file in exclude_files:
                if file in include_files:
                    include_files.remove(file)
                    print(f"<LOADCOMMENT> Removed {file.split('/')[-1]} from training")
        
        dataset = load_dataset(
            'parquet',
            data_files={'train': include_files},
            streaming=True
        ).with_format('numpy')

        return dataset, include_files

class LoaderHF(LoaderObj):
    def __init__(self, 
        train_dataset_path: str,
        train_name: str=None,
        val_dataset_path: str=None,
        val_name: str=None,
        dictionary_path: str=None,
        synonyms: list=None,
        masses_path: str=None,
        tokenizer_path: str=None,
        test_split_method: str='full_val',
        top_pks: int=100,
        batch_size: int=100,
        num_workers: int=0,
        **kwargs
    ):

        dpe = "parquet/processed"

        if val_dataset_path is None:
            val_dataset_path = train_dataset_path
        if masses_path is None:
            masses_path = train_dataset_path
        tokenizer_path = train_dataset_path if tokenizer_path==None else tokenizer_path
        max_seq = kwargs['pep_length'][1] if 'pep_length' in kwargs.keys() else None
        
        ##############
        # Dictionary #
        ##############
        if dictionary_path is not None:
            self.amod_dic = self.create_sequence_dictionary(dictionary_path)
            if synonyms is not None:
                for pair in synonyms:
                    letter_a, letter_b = pair
                    self.amod_dic = self.synonym(letter_a, letter_b, self.amod_dic)
            self.amod_dic_rev = self.reverse_dictionary(self.amod_dic)
        
        #####################
        # Dictionary masses #
        #####################
        # - RULES
        #   1. There is a file that matches the regex *masses.tsv in the masses_path
        self.massdic = self.load_token_masses(masses_path)

        ###############
        # Split sizes #
        ###############
        # - RULES
        #   1. There is a file that matches the regex *sizes.tsv in the train_dataset_path and val_dataset_path
        #   2. val_name will pick out 1 file's size from the val_dataset_path
        self.train_size = self.find_set_size_for_tqdm(train_dataset_path, train_name, val_name, "*species*size*tsv")
        self.val_size = self.find_set_size_for_tqdm(val_dataset_path, val_name, regex="*species*size*tsv")
        
        ###########
        # Dataset #
        ###########
        # - RULES
        #   1. The *_directory_path will contain its data in a directory named "parquet/processed"
        #   2. val_name only has to be somewhere in the filename -> *val_name*
        dataset, train_files = self._load_dataset(join(train_dataset_path, dpe), train_name, val_name, ext='parquet')
        dataset_val, val_files = self._load_dataset(join(val_dataset_path, dpe), val_name)

        print(f"<LOADCOMMENT> Found {len(train_files)} file(s) for training")
        print(f"<LOADCOMMENT> Found {len(val_files)} file(s) for validation")
        
        dataset['val'] = dataset_val['train']

        #########################
        # Map to format outputs #
        #########################
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

        ########################
        # Create test from val #
        ########################
        if test_split_method == 'full_val':
            dataset['test'] = dataset['val']
        elif test_split_method == 'every_other':
            dataset['val'] = dataset['val'].filter(lambda example, idx: idx % every_n == 0, with_indices=True)
            dataset['test'] = dataset['test'].filter(lambda example, idx: idx % every_n == 1, with_indices=True)
        
        #############
        # Tokenizer #
        #############
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        self.tokenizer = self.create_tokenizer(tokenizer_path)
        
        #############
        # Filtering #
        #############
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
        
        # Chimeric
        #dataset = dataset.filter(lambda example: example['chimeric'])

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

        ###############
        # Dataloaders #
        ###############
        num_workers = min(self.dataset['train'].n_shards, num_workers)
        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, num_workers, collate_fn),
            'val':   self.build_dataloader(dataset['val']  , batch_size, 0, collate_fn),
            'test':  self.build_dataloader(dataset['test'] , batch_size, 0, collate_fn),
        }

class LoaderCls(LoaderObj):
    def __init__(
        self,
        dataset_path: str,
        dictionary_path: str=None,
        tokenizer_path: str=None,
        num_workers: int=8,
        batch_size: int=100,
        **kwargs
    ):
        tokenizer_path = dataset_path if tokenizer_path==None else tokenizer_path
        
        # Dictionary
        if dictionary_path is not None:
            self.amod_dic, self.amod_dicr = self.create_sequence_dictionary(dictionary_path)

        # Class mapping
        self.label_dict, self.label_dictr = self.create_label_dictionary(dataset_path)
        
        # Tokenizer
        self.tokenizer = self.create_tokenizer(tokenizer_path)

        # Load dataset
        data_files = join(dataset_path, "parquet/labeled_sequences", "*parquet")
        dataset = load_dataset(
            'parquet',
            data_files=data_files,
        )

        def map_fn_local(example, tokenizer, dic, max_seq):
            tokenized_sequence = tokenizer(example['modified_sequence'])
            length = len(tokenized_sequence)
            full_seq = tokenized_sequence + (max_seq-length) * ['X']
            intseq = [dic[a] for a in full_seq[:max_seq]]
            
            example['intseq'] = intseq
            example['label'] = self.label_dict[example['experiment_name']]
            #example['label'] = example['peptide_length']-1

            return example

        # Map
        lambda_function = lambda example: map_fn_local(
            example,
            tokenizer=self.tokenizer,
            dic=self.amod_dic,
            max_seq=kwargs['pep_length'][1],
        )
        dataset = dataset.map(
            lambda_function, 
            remove_columns=['modified_sequence', '__index_level_0__'],
        )

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (example['peptide_length'] >= kwargs['pep_length'][0]) &
                (example['peptide_length'] <= kwargs['pep_length'][1])
            )

        #dataset = dataset.shuffle()
        #dataset = dataset.flatten_indices()
        dataset = dataset['train'].train_test_split(test_size=0.1)
        
        self.dataset = dataset

        def local_collate_fn(batch_list):
            intseq = th.stack([th.tensor(m['intseq'], dtype=th.int32) for m in batch_list])
            labels = th.cat([th.tensor([m['label']], dtype=th.int32) for m in batch_list])

            return {'intseq': intseq, 'labels': labels}

        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, 0, local_collate_fn, shuffle=True),
            'test': self.build_dataloader(dataset['test'], batch_size, 0, local_collate_fn),
        }

    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.EOS

        return intseq

