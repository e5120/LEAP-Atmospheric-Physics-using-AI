import torch
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, df, feat_columns, label_columns, stage="train"):
        assert stage in ["train", "val", "test"]
        self.df = df
        self.x_df = torch.from_numpy(df[feat_columns].to_numpy())
        if stage != "test":
            self.y_df = torch.from_numpy(df[label_columns].to_numpy())
        self.stage = stage

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, index):
        data = {"input": self.x_df[index]}
        if self.stage in ["train", "val"]:
            data["labels"] = self.y_df[index]
        return data
