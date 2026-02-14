from functools import partial
from glob import glob
import importlib.util
import os
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset

import utils
from loader import LoaderObj, map_fn, collate_fn

join = os.path.join


def _load_tokenizer_no_sys_path(tokenizer_path):
    mod_path = join(tokenizer_path, "enumerate_tokens.py")
    if os.path.exists(mod_path):
        spec = importlib.util.spec_from_file_location("enumerate_tokens_lance", mod_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "partition_modified_sequence", None)
            if fn is not None:
                return fn
    return utils.partition_modified_sequence


def _build_lance_loader(
    dataset,
    batch_size,
    num_workers,
    collate_function,
    drop_last=False,
    shuffle=False,
    sampler=None,
    pin_memory=False,
    persistent_workers=False,
):
    kwargs = {
        "drop_last": drop_last,
        "collate_fn": collate_function,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle

    if num_workers <= 0:
        return DataLoader(dataset, batch_size=batch_size, num_workers=0, **kwargs)

    from lance.torch.data import get_safe_loader
    return get_safe_loader(
        dataset,
        batch_size=batch_size,
        num_workers=int(num_workers),
        persistent_workers=bool(persistent_workers),
        **kwargs,
    )


class LanceMappedDataset(Dataset):
    def __init__(
        self,
        uri,
        tokenizer_path,
        token_dict,
        top_pks=100,
        max_seq=40,
        reverse=False,
    ):
        from lance.torch.data import SafeLanceDataset

        self.uri = uri
        # Keep only length in parent process; open the dataset lazily per process.
        self.length = len(SafeLanceDataset(uri))
        self.dataset = None
        self.dataset_pid = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.tokenizer_pid = None
        self.token_dict = token_dict
        self.pad_token = int(token_dict["X"])
        self.top_pks = top_pks
        self.max_seq = max_seq
        self.reverse = reverse

    def __len__(self):
        return self.length

    def _get_dataset(self):
        pid = os.getpid()
        if self.dataset is None or self.dataset_pid != pid:
            from lance.torch.data import SafeLanceDataset
            self.dataset = SafeLanceDataset(self.uri)
            self.dataset_pid = pid
        return self.dataset

    def _get_tokenizer(self):
        pid = os.getpid()
        if self.tokenizer is None or self.tokenizer_pid != pid:
            self.tokenizer = _load_tokenizer_no_sys_path(self.tokenizer_path)
            self.tokenizer_pid = pid
        return self.tokenizer

    def __getitem__(self, index):
        dataset = self._get_dataset()
        tokenizer = self._get_tokenizer()
        row = dict(dataset[index])
        row["mz_array"] = np.asarray(row["mz_array"], dtype=np.float32)
        row["intensity_array"] = np.asarray(row["intensity_array"], dtype=np.float32)

        if "name" not in row and "experiment_name" in row:
            row["name"] = row["experiment_name"]

        out = map_fn(
            row,
            tokenizer=tokenizer,
            dic=self.token_dict,
            top=self.top_pks,
            max_seq=self.max_seq,
            reverse=self.reverse,
            row_idx=None,
        )

        if "tokenized_sequence" in out and self.max_seq is not None:
            seq = np.asarray(out["tokenized_sequence"], dtype=np.int32)
            peptide_length = int(out.get("peptide_length", len(seq)))
            if peptide_length > self.max_seq:
                peptide_length = self.max_seq
                seq = seq[: self.max_seq]
            if len(seq) < self.max_seq:
                pad = np.full(self.max_seq - len(seq), self.pad_token, dtype=np.int32)
                seq = np.concatenate([seq, pad], axis=0)
            out["peptide_length"] = peptide_length
            out["tokenized_sequence"] = seq

        if "experiment_name" not in out:
            out["experiment_name"] = out.get("name", f"row_{index}")
        return out


class LoaderLance(LoaderObj):
    def __init__(
        self,
        train_dataset_path: str,
        train_name: str = None,
        val_dataset_path: str = None,
        val_name: str = None,
        dictionary_path: str = None,
        synonyms: list = None,
        masses_path: str = None,
        tokenizer_path: str = None,
        test_split_method: str = "full_val",
        top_pks: int = 100,
        pep_length: list = [0, 40],
        reverse: bool = False,
        batch_size: int = 100,
        val_batch_size: int = None,
        test_batch_size: int = None,
        num_workers: int = 0,
        custom_columns: list = [],
        **kwargs,
    ):
        self.custom_columns = []
        self.precomputed_encoder = None
        self.train_sampler = None

        if val_dataset_path is None:
            val_dataset_path = train_dataset_path
        if masses_path is None:
            masses_path = train_dataset_path
        tokenizer_path = train_dataset_path if tokenizer_path is None else tokenizer_path
        max_seq = pep_length[1] if pep_length is not None else None
        val_batch_size = batch_size if val_batch_size is None else val_batch_size
        test_batch_size = val_batch_size if test_batch_size is None else test_batch_size

        if dictionary_path is not None:
            self.amod_dic = self.create_sequence_dictionary(dictionary_path)
            if synonyms is not None:
                for pair in synonyms:
                    letter_a, letter_b = pair
                    self.amod_dic = self.synonym(letter_a, letter_b, self.amod_dic)
            self.amod_dic_rev = self.reverse_dictionary(self.amod_dic)

        self.massdic = self.load_token_masses(masses_path)
        lance_species_root = kwargs.get("lance_species_root")
        train_lance_path = kwargs.get("train_lance_path")
        val_lance_path = kwargs.get("val_lance_path")
        test_lance_path = kwargs.get("test_lance_path")

        if lance_species_root is not None:
            species_paths = sorted(glob(join(lance_species_root, "*.lance")))
            if len(species_paths) == 0:
                raise FileNotFoundError(f"No .lance files found in {lance_species_root}")
            if val_name is None:
                raise ValueError(
                    "loader.val_name is required when using loader.lance_species_root."
                )

            val_paths = [
                fp for fp in species_paths
                if val_name.lower() in os.path.basename(fp).replace(".lance", "").lower()
            ]
            if len(val_paths) == 0:
                raise ValueError(
                    f"No lance species file matched val_name='{val_name}' in {lance_species_root}"
                )
            train_paths = [fp for fp in species_paths if fp not in val_paths]
            if len(train_paths) == 0:
                raise ValueError("No training species remained after selecting validation species.")
        else:
            train_paths = [train_lance_path or train_dataset_path]
            val_paths = [val_lance_path or val_dataset_path]

        if test_lance_path is not None:
            test_paths = [test_lance_path]
        elif test_split_method == "full_val":
            test_paths = list(val_paths)
        else:
            test_paths = list(val_paths)

        print(f"<LOADCOMMENT> Found {len(train_paths)} lance train file(s)")
        print(f"<LOADCOMMENT> Found {len(val_paths)} lance val file(s)")

        train_datasets = [
            LanceMappedDataset(
                uri=fp,
                tokenizer_path=tokenizer_path,
                token_dict=self.amod_dic,
                top_pks=top_pks,
                max_seq=max_seq,
                reverse=reverse,
            )
            for fp in train_paths
        ]
        val_datasets = [
            LanceMappedDataset(
                uri=fp,
                tokenizer_path=tokenizer_path,
                token_dict=self.amod_dic,
                top_pks=top_pks,
                max_seq=max_seq,
                reverse=reverse,
            )
            for fp in val_paths
        ]
        test_datasets = [
            LanceMappedDataset(
                uri=fp,
                tokenizer_path=tokenizer_path,
                token_dict=self.amod_dic,
                top_pks=top_pks,
                max_seq=max_seq,
                reverse=reverse,
            )
            for fp in test_paths
        ]

        train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
        test_dataset = test_datasets[0] if len(test_datasets) == 1 else ConcatDataset(test_datasets)

        debug_subset = kwargs.get("debug_subset")
        if debug_subset is not None:
            debug_subset = int(debug_subset)
            train_dataset = Subset(train_dataset, range(min(debug_subset, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(debug_subset, len(val_dataset))))
            test_dataset = Subset(test_dataset, range(min(debug_subset, len(test_dataset))))

        world_size = utils.get_world_size()
        rank = utils.get_rank()
        self.train_size = int(np.ceil(len(train_dataset) / max(world_size, 1)))
        self.val_size = int(np.ceil(len(val_dataset) / max(world_size, 1)))

        train_sampler = None
        val_sampler = None
        test_sampler = None
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        self.train_sampler = train_sampler

        pin_memory = bool(kwargs.get("pin_memory", False))
        persistent_workers = bool(kwargs.get("persistent_workers", False))
        val_num_workers = int(kwargs.get("val_num_workers", 0))

        train_collate = partial(
            collate_fn,
            custom_columns=[],
            precomputed_encoder=None,
            split_name="train",
        )
        val_collate = partial(
            collate_fn,
            custom_columns=custom_columns,
            precomputed_encoder=None,
            split_name="val",
        )
        test_collate = partial(
            collate_fn,
            custom_columns=custom_columns,
            precomputed_encoder=None,
            split_name="test",
        )

        self.dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        self.dataloader = {
            "train": _build_lance_loader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_function=train_collate,
                drop_last=(world_size > 1),
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
            "val": _build_lance_loader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                collate_function=val_collate,
                drop_last=False,
                shuffle=False,
                sampler=val_sampler,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
            "test": _build_lance_loader(
                dataset=test_dataset,
                batch_size=test_batch_size,
                num_workers=val_num_workers,
                collate_function=test_collate,
                drop_last=False,
                shuffle=False,
                sampler=test_sampler,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
        }

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(int(epoch))
