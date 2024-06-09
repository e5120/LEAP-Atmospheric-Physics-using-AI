from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import leap.dataset
from leap.utils import normalize


class LeapDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.cfg = cfg
        self.iteration = 0
        self.chunk_size = cfg.chunk_size
        self.val_chunk_size = cfg.val_chunk_size
        self.data_dir = Path(cfg.dir.data_dir)
        self.dataset_cls = getattr(leap.dataset, self.cfg.dataset.name)
        # load dataset
        weight_df = pl.read_csv(Path(self.data_dir, "sample_submission.csv"), n_rows=1)[:, 1:]
        sample_df = pl.read_csv(Path(self.data_dir, "train.csv"), n_rows=1)[:, 1:]
        sample_df = sample_df.select(pl.exclude(weight_df.columns))
        if cfg.used_output_cols:
            weight_df = weight_df.select([
                pl.col(f"^{col}.*$")
                for col in cfg.used_output_cols
            ])
        elif cfg.unused_output_cols:
            raise NotImplementedError
        self.label_columns = weight_df.columns
        if cfg.used_input_cols:
            sample_df = sample_df.select([
                pl.col(f"^{col}.*$")
                for col in cfg.used_input_cols
            ])
        elif cfg.unused_input_cols:
            raise NotImplementedError
        self.feature_columns = sample_df.columns
        print(f"# of input size: {len(self.feature_columns)}, # of output size: {len(self.label_columns)}")
        if cfg.stage == "train":
            files = list(self.data_dir.glob("processed_train*.parquet"))
            np.random.shuffle(files)
            self.trn_files = files[: -self.val_chunk_size]
            self.val_files = files[-self.val_chunk_size:]
            dfs = []
            for filename in self.val_files:
                df = pl.read_parquet(filename)
                dfs.append(df)
            df = pl.concat(dfs)
            self.val_df = normalize(df, self.feature_columns, self.label_columns, cfg.scaler, self.data_dir)
            print(f"# of train files: {len(self.trn_files)}, # of val files: {len(self.val_files)}")
            self.val_dataset = None
            print(f"# of val: {len(self.val_df)}")
        else:
            self.test_df = pl.read_parquet(Path(cfg.dir.data_dir, "processed_test.parquet"))
            self.test_df = normalize(self.test_df, self.feature_columns, None, cfg.scaler, self.data_dir)
            self.test_dataset = None
            print(f"# of test: {len(self.test_df)}")

    def _generate_dataset(self, stage):
        if self.iteration == 0 and stage == "train":
            np.random.shuffle(self.trn_files)
        if stage == "train":
            dfs = []
            for i in tqdm(range(self.chunk_size)):
                df = pl.read_parquet(self.trn_files[self.iteration*self.chunk_size+i])
                dfs.append(df)
            df = pl.concat(dfs)
            df = normalize(df, self.feature_columns, self.label_columns, self.cfg.scaler, self.data_dir)
            self.iteration += 1
            if (self.iteration + 1) * self.chunk_size >= len(self.trn_files):
                self.iteration = 0
        elif stage == "val":
            if self.val_dataset is not None:
                return self.val_dataset
            df = self.val_df
        elif stage == "test":
            if self.test_dataset is not None:
                return self.test_dataset
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(df, self.feature_columns, self.label_columns, stage=stage, **self.cfg.dataset.params)
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
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")
