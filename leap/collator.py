import numpy as np
import torch

from leap.utils import IN_VECTOR_COLUMNS, IN_COLUMNS, IN_AUX_COLUMNS, OUT_COLUMNS


def get_auxiliary_data(batch):
    aux = []
    for col in IN_AUX_COLUMNS:
        aux.append(
            torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
        )
    aux = torch.concat(aux, dim=1)
    return aux


def simple_collate_fn(batch, stage):
    for col in IN_COLUMNS:
        batch[col] = torch.from_numpy(batch[col])
    if stage != "test":
        labels = []
        for col in OUT_COLUMNS:
            batch[col] = torch.from_numpy(batch[col]).reshape(batch[col].shape[0], -1)
            labels.append(batch[col])
        labels = torch.concat(labels, dim=1)
        batch["labels"] = labels
    return batch


def single_collate_fn(batch, stage):
    inputs = []
    for col in IN_COLUMNS:
        inputs.append(
            torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
        )
    inputs = torch.concat(inputs, dim=1)
    aux = get_auxiliary_data(batch)
    if stage != "test":
        labels = []
        for col in OUT_COLUMNS:
            labels.append(
                torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
            )
        labels = torch.concat(labels, dim=1)
        return {
            "input": inputs,
            "labels": labels,
            "aux": aux,
        }
    else:
        return {
            "input": inputs,
            "aux": aux,
        }


def sequential_collate_fn(batch, stage):
    scalars = []
    vectors = []
    for col in IN_COLUMNS:
        if col in IN_VECTOR_COLUMNS:
            vectors.append(batch[col])
        else:
            scalars.append(batch[col])
    scalars = torch.from_numpy(np.stack(scalars, axis=1))
    vectors = torch.from_numpy(np.stack(vectors, axis=1))
    aux = get_auxiliary_data(batch)
    if stage != "test":
        labels = []
        for col in OUT_COLUMNS:
            labels.append(
                torch.from_numpy(batch[col].reshape(batch[col].shape[0], -1))
            )
        labels = torch.concat(labels, dim=1)
        return {
            "x_scalar": scalars,
            "x_vector": vectors,
            "labels": labels,
            "aux": aux,
        }
    else:
        return {
            "x_scalar": scalars,
            "x_vector": vectors,
            "aux": aux,
        }
