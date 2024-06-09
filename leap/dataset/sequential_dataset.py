import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


SCALAR_COLUMNS = [
    "state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX", "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS",
    "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR", "cam_in_LWUP", "cam_in_ICEFRAC", "cam_in_LANDFRAC", "cam_in_OCNFRAC", "cam_in_SNOWHLAND",
]
VECTOR_COLUMNS = [
    "state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O",
]


class SequentialDataset(Dataset):
    def __init__(self, df, feat_columns, label_columns, stage="train"):
        assert stage in ["train", "val", "test"]
        self.df = df.select("sample_id")
        self.scalar_arr = torch.from_numpy(np.array(df[SCALAR_COLUMNS].to_numpy().tolist()))
        for col in VECTOR_COLUMNS:
            df = df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
        self.vector_arr = torch.from_numpy(np.array(df[VECTOR_COLUMNS].to_numpy().tolist()))
        if stage != "test":
            self.y_arr = torch.from_numpy(np.array(df[label_columns].to_numpy().tolist()))
        self.stage = stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = {
            "scalar": self.scalar_arr[index],
            "vector": self.vector_arr[index],
        }
        if self.stage in ["train", "val"]:
            data["labels"] = self.y_arr[index]
        return data
