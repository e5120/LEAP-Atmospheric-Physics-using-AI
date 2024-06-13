import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from leap.dataset import BaseDataset
from leap.utils import IN_SCALAR_COLUMNS, IN_VECTOR_COLUMNS, OUT_SCALAR_COLUMNS, OUT_VECTOR_COLUMNS


class SequentialDataset(BaseDataset):
    def __init__(self, df, feat_columns, label_columns, stage="train"):
        super().__init__(df, feat_columns, label_columns, stage=stage)
        if stage != "test":
            self.y_arr = torch.concat([self.y_scalar_arr, self.y_vector_arr.reshape(self.x_vector_arr.size(0), -1)], dim=1)

    def __getitem__(self, index):
        data = {
            "x_scalar": self.x_scalar_arr[index],
            "x_vector": self.x_vector_arr[index],
        }
        if self.stage in ["train", "val"]:
            data["labels"] = self.y_arr[index]
        return data
