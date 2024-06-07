from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"].float())
        output["loss"] = loss
        return output
