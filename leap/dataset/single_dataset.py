from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, df, feat_columns, label_columns, stage="train"):
        assert stage in ["train", "val", "test"]
        self.x_df = df[feat_columns].to_numpy()
        self.y_df = df[label_columns].to_numpy()
        self.stage = stage

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, index):
        data = {"input": self.x_df[index]}
        if self.stage in ["train", "val"]:
            data["labels"] = self.y_df[index]
        return data
