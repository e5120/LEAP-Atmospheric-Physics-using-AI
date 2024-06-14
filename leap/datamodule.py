from pathlib import Path

import lightning as L

import leap.collator
from leap.tfrecord import TFRecordDataLoader
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
        self.batch_size = cfg.batch_size
        self.data_dir = Path(cfg.dir.data_dir)
        self.collate_fn = getattr(leap.collator, cfg.model.collate_fn)
        self.collate_params = cfg.model.collate_params
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
        print(f"# of input size: {len(self.input_size)}, # of output size: {len(self.output_size)}")

    def train_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="train")

    def val_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="val")

    def test_dataloader(self):
        return TFRecordDataLoader(self.data_dir, batch_size=self.batch_size, stage="test")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return self.collate_fn(batch, self.feature_columns, self.label_columns, **self.collate_params)

    @property
    def input_size(self):
        vec_feats = list(filter(lambda x: x in IN_VECTOR_COLUMNS, self.feature_columns))
        return len(self.feature_columns) - len(vec_feats) + len(vec_feats) * 60

    @property
    def output_size(self):
        vec_feats = list(filter(lambda x: x in OUT_VECTOR_COLUMNS, self.label_columns))
        return len(self.label_columns) - len(vec_feats) + len(vec_feats) * 60
