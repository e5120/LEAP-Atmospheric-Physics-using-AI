import torch


def single_collate_fn(batch, feat_cols, label_cols):
    inputs = []
    for col in feat_cols:
        inputs.append(
            torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
        )
    inputs = torch.concat(inputs, dim=1)
    labels = []
    for col in label_cols:
        labels.append(
            torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
        )
    labels = torch.concat(labels, dim=1)
    return {
        "input": inputs,
        "labels": labels,
    }
