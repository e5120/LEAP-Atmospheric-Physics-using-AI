from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import leap.dataset
from leap.utils import (
    IN_COLUMNS,
    IN_SCALAR_COLUMNS,
    IN_VECTOR_COLUMNS,
    OUT_COLUMNS,
    OUT_SCALAR_COLUMNS,
    OUT_VECTOR_COLUMNS,
)


class LeapDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.iteration = 0
        self.chunk_size = cfg.chunk_size
        self.data_dir = Path(cfg.dir.data_dir)
        self.dataset_cls = getattr(leap.dataset, self.cfg.dataset.name)
        # load dataset
        if cfg.used_output_cols:
            self.label_columns = list(filter(lambda x: x in cfg.used_output_cols), OUT_COLUMNS)
        elif cfg.unused_output_cols:
            self.label_columns = list(filter(lambda x: x not in cfg.used_output_cols), OUT_COLUMNS)
        else:
            self.label_columns = OUT_COLUMNS
        if cfg.used_input_cols:
            self.feature_columns = list(filter(lambda x: x in cfg.used_input_cols), IN_COLUMNS)
        elif cfg.unused_input_cols:
            self.feature_columns = list(filter(lambda x: x not in cfg.used_input_cols), IN_COLUMNS)
        else:
            self.feature_columns = IN_COLUMNS
        print(f"# of input size: {len(self.feature_columns)}, # of output size: {len(self.label_columns)}")
        if cfg.stage == "train":
            files = sorted(list(self.data_dir.glob("processed_train*.parquet")))
            self.trn_files = files[: -cfg.val_chunk_size]
            self.val_files = files[-cfg.val_chunk_size:]
            print(f"# of train files: {len(self.trn_files)}, # of val files: {len(self.val_files)}")
            dfs = []
            for filename in self.val_files:
                df = pl.read_parquet(filename, columns=["sample_id"]+self.feature_columns+self.label_columns)
                dfs.append(df)
            self.val_df = pl.concat(dfs)
            self.val_dataset = self._generate_dataset("val")
            print(f"# of val: {len(self.val_dataset)}")
        else:
            self.test_df = pl.read_parquet(Path(cfg.dir.data_dir, "processed_test.parquet"), columns=["sample_id"]+self.feature_columns)
            self.test_dataset = self._generate_dataset("test")
            print(f"# of test: {len(self.test_dataset)}")

    def _generate_dataset(self, stage):
        if stage == "train":
            dfs = []
            for i in tqdm(range(self.chunk_size)):
                j = (self.iteration * self.chunk_size + i) % len(self.trn_files)
                if j == 0:
                    np.random.shuffle(self.trn_files)
                df = pl.read_parquet(self.trn_files[j], columns=["sample_id"]+self.feature_columns+self.label_columns)
                dfs.append(df)
            df = pl.concat(dfs)
            self.iteration += 1
        elif stage == "val":
            if hasattr(self, "val_dataset"):
                return self.val_dataset
            df = self.val_df
        elif stage == "test":
            if hasattr(self, "test_dataset"):
                return self.test_dataset
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(df, self.feature_columns, self.label_columns, precision=self.cfg.trainer.precision, stage=stage, **self.cfg.dataset.params)
        if stage == "val":
            self.val_dataset = dataset
        elif stage == "test":
            self.test_dataset = dataset
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "train":
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            # persistent_workers=True,
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")

    @property
    def input_size(self):
        vec_feats = list(filter(lambda x: x in IN_VECTOR_COLUMNS, self.feature_columns))
        return len(self.feature_columns) - len(vec_feats) + len(vec_feats) * 60

    @property
    def output_size(self):
        vec_feats = list(filter(lambda x: x in OUT_VECTOR_COLUMNS, self.label_columns))
        return len(self.label_columns) - len(vec_feats) + len(vec_feats) * 60
