import torch
import torch.nn as nn

from leap.model import BaseModel
from leap.model.modules import MLPBlock


class FFNModelV2(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1, activation="relu",
                 num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        input_size = 2 * num_vector_feats * 59 + 3 * num_vector_feats + num_scalar_feats
        self.layers = MLPBlock(
            input_size, output_size,
            hidden_sizes=hidden_sizes,
            p=dropout,
            activation=activation,
        )

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        scalar = batch["x_scalar"]
        vec_diff1 = batch["x_vector"].diff(dim=-1).reshape(bs, -1)
        vec_diff2 = batch["x_vector"].flip(dims=[-1]).diff(dim=-1).reshape(bs, -1)
        vec_min = torch.min(batch["x_vector"], dim=-1)[0]
        vec_mean = torch.mean(batch["x_vector"], dim=-1)
        vec_max = torch.max(batch["x_vector"], dim=-1)[0]
        inputs = torch.cat([scalar, vec_diff1, vec_diff2, vec_min, vec_mean, vec_max], dim=1)
        logits = self.layers(inputs)
        return {
            "logits": logits,
        }
