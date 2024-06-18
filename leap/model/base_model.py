from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, ignore_mask=None):
        super().__init__()
        self.ignore_mask = ignore_mask
        self.loss_fn = nn.MSELoss(reduction="none")

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"])
        if self.ignore_mask is not None:
            loss = loss[:, self.ignore_mask]
        loss = loss.mean()
        output["loss"] = loss
        return output
