from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re
from glob import glob
import sys
import pandas as pd
import numpy as np
import math
join = os.path.join

def map_fn(example, tokenizer, dic=None, top=100, max_seq=50, reverse=False, row_idx=None):
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
    example['spectrum_length'] = spectrum_length #len(example['mz_array'])
    if 'modified_sequence' in example:
        tokenized_sequence = tokenizer(example['modified_sequence'])
        peptide_length = len(tokenized_sequence)
        if reverse:
            tokenized_sequence = tokenized_sequence[::-1]
        example['tokenized_sequence'] = np.array([dic.get(m, dic['X']) for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=np.int32)
        example['peptide_length'] = peptide_length
    if 'name' in example: example['experiment_name'] = example['name'] # compat
    if row_idx is not None:
        example['row_idx'] = int(row_idx)

    return example

def collate_fn(batch_list, custom_columns=[], precomputed_encoder=None, split_name=None):
    out = {}
    out['experiment_name'] = np.array([m['experiment_name'] for m in batch_list])
    out['length'] = th.tensor(np.stack([m['spectrum_length'] for m in batch_list]), dtype=th.int32)
    maxlength = out['length'].max()
    out['mz'] = th.tensor(np.stack([m['mz_array'][:maxlength] for m in batch_list]), dtype=th.float32)
    out['ab'] = th.tensor(np.stack([m['intensity_array'][:maxlength] for m in batch_list]), dtype=th.float32)
    out['charge'] = th.tensor(np.stack([m['precursor_charge'] for m in batch_list]), dtype=th.int32)
    out['mass'] = th.tensor(np.stack([m['precursor_mass'] for m in batch_list]), dtype=th.float32)
    if 'tokenized_sequence' in batch_list[0].keys():
        out['peplen'] = th.tensor(np.stack([m['peptide_length'] for m in batch_list]), dtype=th.int32)
        out['intseq'] = th.tensor(np.stack([m['tokenized_sequence'][:out['peplen'].max()] for m in batch_list]), dtype=th.int32)
        #out['intseq'] = th.tensor(np.stack([m['tokenized_sequence'][:41] for m in batch_list]), dtype=th.int32)
    if 'chimeric' in batch_list[0].keys():
        out['chimeric'] = th.tensor(np.stack([m['chimeric'] for m in batch_list]))
    if 'Hyperscore' in batch_list[0]:
        out['hyperscore'] = th.tensor(np.stack([m['Hyperscore'] for m in batch_list]))
    for column in custom_columns:
        out[column] = np.array([m[column] for m in batch_list])
    if (
        precomputed_encoder is not None and
        split_name is not None and
        split_name in precomputed_encoder and
        'row_idx' in batch_list[0]
    ):
        row_idx = np.array([m['row_idx'] for m in batch_list], dtype=np.int64)
        split_store = precomputed_encoder[split_name]
        out['enc_emb'] = th.tensor(split_store['emb'][row_idx], dtype=th.float32)
        out['enc_mask'] = th.tensor(split_store['mask'][row_idx], dtype=th.bool)

    return out

class LoaderObj:
    def build_dataloader(self, dataset, batch_size, num_workers, collate_fn, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
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
        low = int(np.minimum(amod_dic[token1], amod_dic[token2]))
        high = int(np.maximum(amod_dic[token1], amod_dic[token2]))
        amod_dic[token1] = amod_dic[token2] = low
        amod_dic = {key:value if value < high else value-1 for key, value in amod_dic.items()}
        return amod_dic
    
    def create_label_dictionary(self, dataset_path, filename="species_list.txt"):
        filepath = join(dataset_path, f"parquet/labeled_sequences/{filename}")
        List = open(filepath).read().strip().split("\n")
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
        try:
            from enumerate_tokens import partition_modified_sequence
            tokenizer = partition_modified_sequence
        except ImportError:
            tokenizer = utils.partition_modified_sequence

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
        ss_matches = glob(join(dataset_path, regex))
        if len(ss_matches) == 0:
            return float('inf')
        ss_path = ss_matches[0]
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

    def _load_dataset(self, dataset_path, include_name=None, exclude_name=None, ext=None, streaming=True, keep_in_memory=False):
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
            streaming=streaming,
            keep_in_memory=(keep_in_memory if not streaming else False),
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
        pep_length: list=[0,40],
        reverse: bool=False,
        batch_size: int=100,
        val_batch_size: int=None,
        test_batch_size: int=None,
        num_workers: int=0,
        custom_columns: list=[],
        **kwargs
    ):

        dpe = "parquet/processed"
        self.custom_columns = []

        if val_dataset_path is None:
            val_dataset_path = train_dataset_path
        if masses_path is None:
            masses_path = train_dataset_path
        tokenizer_path = train_dataset_path if tokenizer_path==None else tokenizer_path
        max_seq = pep_length[1] if pep_length is not None else None
        val_batch_size = batch_size if val_batch_size is None else val_batch_size
        test_batch_size = val_batch_size if test_batch_size is None else test_batch_size
        streaming = kwargs.get('streaming', True)
        keep_in_memory = kwargs.get('keep_in_memory', False)
        split_parquets = kwargs.get('split_parquets', False)
        use_precomputed_encoder = kwargs.get('use_precomputed_encoder', False)
        encoder_arrays_path = kwargs.get('encoder_arrays_path', train_dataset_path)
        
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
        if split_parquets:
            train_fp = join(train_dataset_path, "train.parquet")
            val_fp = join(val_dataset_path, "val.parquet")
            test_fp = join(val_dataset_path, "test.parquet")
            self.train_size = len(pd.read_parquet(train_fp, columns=[]))
            self.val_size = len(pd.read_parquet(val_fp, columns=[]))
        else:
            self.train_size = self.find_set_size_for_tqdm(train_dataset_path, train_name, val_name, "*species*size*tsv")
            self.val_size = self.find_set_size_for_tqdm(val_dataset_path, val_name, regex="*species*size*tsv")
        
        ###########
        # Dataset #
        ###########
        # - RULES
        #   1. The *_directory_path will contain its data in a directory named "parquet/processed"
        #   2. val_name only has to be somewhere in the filename -> *val_name*
        if split_parquets:
            data_files = {'train': train_fp, 'val': val_fp}
            if os.path.exists(test_fp):
                data_files['test'] = test_fp
            dataset = load_dataset(
                'parquet',
                data_files=data_files,
                streaming=streaming,
                keep_in_memory=(keep_in_memory if not streaming else False),
            ).with_format('numpy')
            train_files = [train_fp]
            val_files = [val_fp]
            self.precomputed_encoder = None
            if use_precomputed_encoder:
                self.precomputed_encoder = {}
                for split_name in ['train', 'val', 'test']:
                    emb_fp = join(encoder_arrays_path, f'enc_embs_{split_name}.npy')
                    mask_fp = join(encoder_arrays_path, f'enc_mask_{split_name}.npy')
                    if os.path.exists(emb_fp) and os.path.exists(mask_fp):
                        mmap_mode = None if keep_in_memory else 'r'
                        self.precomputed_encoder[split_name] = {
                            'emb': np.load(emb_fp, mmap_mode=mmap_mode),
                            'mask': np.load(mask_fp, mmap_mode=mmap_mode),
                        }
        else:
            dataset, train_files = self._load_dataset(
                join(train_dataset_path, dpe), train_name, val_name, ext='parquet', streaming=streaming, keep_in_memory=keep_in_memory
            )
            dataset_val, val_files = self._load_dataset(
                join(val_dataset_path, dpe), val_name, streaming=streaming, keep_in_memory=keep_in_memory
            )
            dataset['val'] = dataset_val['train']
            self.precomputed_encoder = None

        print(f"<LOADCOMMENT> Found {len(train_files)} file(s) for training")
        print(f"<LOADCOMMENT> Found {len(val_files)} file(s) for validation")

        # Optional debug subset to speed up iteration
        debug_subset = kwargs.get('debug_subset')
        if debug_subset is not None:
            def subset(ds):
                if isinstance(ds, IterableDataset):
                    return ds.take(int(debug_subset))
                max_n = min(int(debug_subset), len(ds))
                return ds.select(range(max_n))
            dataset['train'] = subset(dataset['train'])
            dataset['val'] = subset(dataset['val'])
            if 'test' in dataset:
                dataset['test'] = subset(dataset['test'])
            if self.train_size not in [None, float('inf')]:
                self.train_size = min(self.train_size, int(debug_subset))
            if self.val_size not in [None, float('inf')]:
                self.val_size = min(self.val_size, int(debug_subset))

        #############
        # Tokenizer #
        #############
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        self.tokenizer = self.create_tokenizer(tokenizer_path)

        #########################
        # Map to format outputs #
        #########################
        lambda_function = lambda example, idx: map_fn(
            example,
            tokenizer=self.tokenizer,
            dic=self.amod_dic,
            top=top_pks, 
            max_seq=max_seq,
            reverse=reverse,
            row_idx=idx if use_precomputed_encoder else None,
        )
        if 'remove_columns' in kwargs:
            remove_train_columns = [column for column in kwargs['remove_columns'] if column in dataset['train'].features]
            remove_val_columns = [column for column in kwargs['remove_columns'] if column in dataset['val'].features]
        else:
            remove_train_columns = []
            remove_val_columns = []
        dataset['train'] = dataset['train'].map(
            lambda_function, 
            remove_columns=remove_train_columns,
            with_indices=True,
        )
        dataset['val'] = dataset['val'].map(
            lambda_function,
            remove_columns=remove_val_columns,
            with_indices=True,
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
        # Filtering #
        #############
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['precursor_charge'] >= kwargs['charge'][0]) &
                (example['precursor_charge'] <= kwargs['charge'][1])
            )

        # Filter for length
        if pep_length is not None:
            dataset = dataset.filter(
                lambda example: 
                (len(example['tokenized_sequence']) >= pep_length[0]) &
                (len(example['tokenized_sequence']) <= pep_length[1])
            )
        
        # Chimeric
        #dataset = dataset.filter(lambda example: example['chimeric'])

        # Filter val set for dispersed examples
        if ('val_steps' in kwargs.keys()) and ('disperse' in kwargs.keys()):
            if kwargs['disperse'] and (kwargs['val_steps'] is not None):
                print(f"<LOADCOMMENT> Dispersing validation set into {kwargs['val_steps']} batches")
                every_n_ = self.val_size // val_batch_size // kwargs['val_steps']
                if every_n_ > 2:
                    every_n_ -= 1 # minus 1 to be safe (charge and length filter make dataset shorter)
                every_n = np.maximum(1, every_n_) 
                dataset['val'] = dataset['val'].filter(lambda example, idx: idx % every_n == 0, with_indices=True)
                self.val_size = kwargs['val_steps'] * val_batch_size
        
        # Shard dataset for distributed training (must occur before shuffle)
        world_size = utils.get_world_size()
        rank = utils.get_rank()
        if world_size > 1:
            def shard_dataset(ds):
                if isinstance(ds, IterableDataset):
                    return ds.filter(lambda _, idx: idx % world_size == rank, with_indices=True)
                return ds.shard(num_shards=world_size, index=rank, contiguous=False)
            dataset['train'] = shard_dataset(dataset['train'])
            if 'val' in dataset:
                dataset['val'] = shard_dataset(dataset['val'])
            if 'test' in dataset:
                dataset['test'] = shard_dataset(dataset['test'])
            if self.train_size not in [None, float('inf')]:
                self.train_size = int(np.ceil(self.train_size / world_size))
            if self.val_size not in [None, float('inf')]:
                self.val_size = int(np.ceil(self.val_size / world_size))

        # Shuffle the dataset
        if 'buffer_size' in kwargs.keys():
            if isinstance(dataset['train'], IterableDataset):
                dataset['train'] = dataset['train'].shuffle(buffer_size=kwargs['buffer_size'])
            else:
                dataset['train'] = dataset['train'].shuffle()
        else:
            dataset['train'] = dataset['train'].shuffle()
        
        self.dataset = dataset

        ###############
        # Dataloaders #
        ###############
        train_n_shards = getattr(self.dataset['train'], "n_shards", None)
        if train_n_shards is not None:
            num_workers = min(train_n_shards, num_workers)
        train_collate_function = lambda x: collate_fn(
            x,
            custom_columns=[],
            precomputed_encoder=self.precomputed_encoder,
            split_name='train',
        )
        val_collate_function = lambda x: collate_fn(
            x,
            custom_columns=custom_columns,
            precomputed_encoder=self.precomputed_encoder,
            split_name='val',
        )
        test_collate_function = lambda x: collate_fn(
            x,
            custom_columns=custom_columns,
            precomputed_encoder=self.precomputed_encoder,
            split_name='test',
        )
        drop_last_train = utils.get_world_size() > 1
        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, num_workers, train_collate_function, drop_last=drop_last_train),
            'val':   self.build_dataloader(dataset['val']  , val_batch_size, 0, val_collate_function),
            'test':  self.build_dataloader(dataset['test'] , test_batch_size, 0, test_collate_function),
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
            self.amod_dic = self.create_sequence_dictionary(dictionary_path)

        # Class mapping
        self.label_dict, self.label_dictr = self.create_label_dictionary(dataset_path, filename="classes.txt")
        
        # Tokenizer
        self.tokenizer = self.create_tokenizer(tokenizer_path)

        # Substitution rates
        #a = {m.split("\t")[0]:float(m.split("\t")[1]) for m in open(join(dataset_path, "parquet/labeled_sequences/subrate.tsv")).read().strip().split("\n")}
        #start=[b.split('->')[0] for b,c in a.items()]
        #end=[b.split('->')[1] for b,c in a.items()]

        # Load dataset
        data_files = join(dataset_path, "parquet/labeled_sequences", "*parquet")
        dataset = load_dataset(
            'parquet',
            data_files=data_files,
        )
        #dataset['train'] = dataset['train'].select(np.arange(0,100, 10))
        dataset['train'] = dataset['train'].shuffle()

        def map_fn_local(example, tokenizer, token_dic, label_dic, max_seq):
            example['peptide_length'] = len(example['sequence'])
            tokenized_sequence = tokenizer(example['sequence'])
            length = len(tokenized_sequence)
            full_seq = np.array(tokenized_sequence + (max_seq-length) * ['X'], dtype='U20')
            ism = full_seq == 'M'
            switch = np.random.rand(len(ism)) < 0.74
            np.put(full_seq, np.where(ism&switch), v='M+15.995')
            isn = full_seq == 'N'
            switch = np.random.rand(len(isn)) < 0.03
            np.put(full_seq, np.where(isn&switch), v='N+0.984')
            isq = full_seq == 'Q'
            switch = np.random.rand(len(isq)) < 0.01
            np.put(full_seq, np.where(isq&switch), v='Q+0.984')
            if np.random.rand() < 0.01:
                nterm = str(np.random.choice(['+42.011', '+43.006', '-17.027']))
                full_seq = np.append(nterm, full_seq[:-1])
            
            intseq = [token_dic[a] for a in full_seq[:max_seq]]
            
            example['intseq'] = intseq
            example['label'] = np.zeros(len(label_dic))
            np.put(example['label'], [label_dic[m] for m in example['enzymes']], 1)
            #print(example['label'])
            #example['label'] = example['peptide_length']-1

            return example

        # Map
        lambda_function = lambda example: map_fn_local(
            example,
            tokenizer=self.tokenizer,
            token_dic=self.amod_dic,
            label_dic=self.label_dict,
            max_seq=kwargs['pep_length'][1],
        )
        dataset = dataset.filter(
            lambda example:
            'U' not in example['sequence']
        )
        dataset = dataset.map(
            lambda_function, 
            remove_columns=['sequence'],
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
