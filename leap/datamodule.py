from pathlib import Path

import lightning as L

import leap.collator
from leap.tfrecord import TFRecordDataLoader


class LeapDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.stage in ["train", "test"]
        self.stage = cfg.stage
        self.batch_size = cfg.batch_size
        self.data_dir = Path(cfg.dir.data_dir, cfg.dataset_name)
        self.num_train_files = cfg.num_train_files
        self.collate_fn = getattr(leap.collator, cfg.model.collate_fn)
        self.collate_params = cfg.model.collate_params

    def train_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="train", num_files=self.num_train_files)

    def val_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="val")

    def test_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="test")

    def predict_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="test")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return self.collate_fn(batch, self.stage, **self.collate_params)
