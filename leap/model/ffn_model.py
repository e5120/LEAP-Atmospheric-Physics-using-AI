import torch.nn as nn

from leap.model import BaseModel
from leap.model.modules import MLPBlock


class FFNModel(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1, activation="swish", ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        self.mlp_layers = MLPBlock(
            input_size, output_size,
            hidden_sizes=hidden_sizes,
            p=dropout,
            activation=activation,
        )

    def forward(self, batch):
        logits = self.mlp_layers(batch["input"])
        return {
            "logits": logits,
        }
