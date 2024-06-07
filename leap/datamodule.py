from pathlib import Path

import numpy as  np
import polars as pl
import lightning as L
from torch.utils.data import DataLoader

import leap.dataset


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
        weight_df = pl.read_csv(Path(self.data_dir, "sample_submission.csv"), n_rows=1)
        sample_df = pl.read_csv(Path(self.data_dir, "train.csv"), n_rows=1)
        self.label_columns = weight_df.columns[1:]
        self.feature_columns = sample_df.select(pl.exclude(self.label_columns)).columns[1:]
        if cfg.stage == "train":
            files = list(self.data_dir.glob("processed_train*.parquet"))
            np.random.shuffle(files)
            self.trn_files = files[: -self.val_chunk_size]
            self.val_files = files[-self.val_chunk_size:]
            dfs = []
            for filename in self.val_files:
                df = pl.read_parquet(filename)
                dfs.append(df)
            self.val_df = pl.concat(dfs)
            print(f"val files: {self.val_files}")
            print(f"# of val: {len(self.val_df)}")
        else:
            self.test_df = pl.read_parquet(Path(cfg.dir.data_dir, "processed_test.parquet"))
            print(f"# of test: {len(self.test_df)}")

    def _generate_dataset(self, stage):
        if self.iteration == 0 and stage == "train":
            np.random.shuffle(self.trn_files)
        if stage == "train":
            dfs = []
            for i in range(self.chunk_size):
                df = pl.read_parquet(self.trn_files[self.iteration*self.chunk_size+i])
                dfs.append(df)
            df = pl.concat(dfs)
            self.iteration += 1
            if self.iteration * self.cfg.chunk_size >= len(self.trn_files):
                self.iteration = 0
        elif stage == "val":
            df = self.val_df
        elif stage == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = self.dataset_cls(df, self.feature_columns, self.label_columns, stage=stage, **self.cfg.dataset.params)
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
