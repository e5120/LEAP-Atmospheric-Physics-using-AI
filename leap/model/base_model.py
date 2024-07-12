from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, ignore_mask=None, use_aux=False, aux_weight=0.1, delta=1.0):
        super().__init__()
        self.ignore_mask = ignore_mask
        self.loss_fn = nn.HuberLoss(reduction="none", delta=delta)
        self.use_aux = use_aux
        if self.use_aux:
            self.aux_weight = aux_weight
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
            loc_logits = self.aux_decoder(output["hidden_state"])
            loc_loss = self.aux_loss_fn(loc_logits, batch["aux"][:, 0].long())
            loss = loss + self.aux_weight * loc_loss
        output["loss"] = loss
        return output
