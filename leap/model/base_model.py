from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, ignore_mask=None, alpha=0.5):
        super().__init__()
        self.ignore_mask = ignore_mask
        self.alpha = alpha
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        # self.mse_loss_fn = nn.MSELoss(reduction="none")
        # self.mae_loss_fn = nn.L1Loss(reduction="none")

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"])
        # mse_loss = self.mse_loss_fn(output["logits"], batch["labels"])
        # mae_loss = self.mae_loss_fn(output["logits"], batch["labels"])
        if self.ignore_mask is not None:
            loss = loss[:, self.ignore_mask]
            # mse_loss = mse_loss[:, self.ignore_mask]
            # mae_loss = mae_loss[:, self.ignore_mask]
        loss = loss.mean()
        # mse_loss = mse_loss.mean()
        # mae_loss = mae_loss.mean()
        # loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        output["loss"] = loss
        return output
