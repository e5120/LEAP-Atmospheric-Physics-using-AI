import numpy as np
import torch

from leap.utils import IN_VECTOR_COLUMNS


def single_collate_fn(batch, feat_cols, label_cols, stage):
    inputs = []
    for col in feat_cols:
        inputs.append(
            torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
        )
    inputs = torch.concat(inputs, dim=1)
    if stage != "test":
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
    else:
        return {
            "input": inputs,
        }


def sequential_collate_fn(batch, feat_cols, label_cols, stage):
    scalars = []
    vectors = []
    for col in feat_cols:
        if col in IN_VECTOR_COLUMNS:
            vectors.append(batch[col])
        else:
            scalars.append(batch[col])
    scalars = torch.from_numpy(np.stack(scalars, axis=1))
    vectors = torch.from_numpy(np.stack(vectors, axis=1))
    if stage != "test":
        labels = []
        for col in label_cols:
            labels.append(
                torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
            )
        labels = torch.concat(labels, dim=1)
        return {
            "x_scalar": scalars,
            "x_vector": vectors,
            "labels": labels,
        }
    else:
        return {
            "x_scalar": scalars,
            "x_vector": vectors,
        }
