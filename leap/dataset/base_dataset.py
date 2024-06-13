from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from leap.utils import IN_SCALAR_COLUMNS, IN_VECTOR_COLUMNS, OUT_SCALAR_COLUMNS, OUT_VECTOR_COLUMNS


class BaseDataset(Dataset):
    def __init__(self, df, feat_columns, label_columns, stage="train"):
        assert stage in ["train", "val", "test"]
        self.df = df.select("sample_id")
        self.in_scalar_cols = list(filter(lambda x: x in IN_SCALAR_COLUMNS, feat_columns))
        self.in_vector_cols = list(filter(lambda x: x in IN_VECTOR_COLUMNS, feat_columns))
        self.in_cols = self.in_scalar_cols + self.in_vector_cols
        self.x_scalar_arr = torch.from_numpy(
            np.array(df[self.in_scalar_cols].to_numpy().tolist())
        )
        self.x_vector_arr = torch.from_numpy(
            np.array(df[self.in_vector_cols].to_numpy().tolist())
        )
        self.out_scalar_cols = list(filter(lambda x: x in OUT_SCALAR_COLUMNS, label_columns))
        self.out_vector_cols = list(filter(lambda x: x in OUT_VECTOR_COLUMNS, label_columns))
        self.label_cols = self.out_scalar_cols + self.out_vector_cols
        if stage != "test":
            self.out_cols = self.out_scalar_cols + self.out_vector_cols
            self.y_scalar_arr = torch.from_numpy(
                np.array(df[self.out_scalar_cols].to_numpy().tolist())
            )
            self.y_vector_arr = torch.from_numpy(
                np.array(df[self.out_vector_cols].to_numpy().tolist())
            )
        self.stage = stage

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
