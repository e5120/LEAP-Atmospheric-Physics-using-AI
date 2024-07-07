from abc import abstractmethod

import torch.nn as nn

from leap.model.modules import MLPBlock
from leap.utils import NUM_GRID


class BaseModel(nn.Module):
    def __init__(self, ignore_mask=None, alpha=0.5, use_aux=False):
        super().__init__()
        self.ignore_mask = ignore_mask
        self.alpha = alpha
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.use_aux = use_aux
        if self.use_aux:
            self.loc_fc = MLPBlock(368, NUM_GRID, hidden_sizes=[512, 512], p=0.1)
            self.aux_loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"])
        if self.ignore_mask is not None:
            loss = loss[:, self.ignore_mask]
        loss = loss.mean()
        if self.use_aux and self.training:
            loc_logits = self.loc_fc(output["logits"])
            loc_loss = self.aux_loss_fn(loc_logits, batch["aux"][:, 0].long())
            loss += 0.1 * loc_loss
        output["loss"] = loss
        return output
