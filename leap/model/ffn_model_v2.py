import torch
import torch.nn as nn

from leap.model import BaseModel


class FFNModelV2(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, num_scalar_feats=16, num_vector_feats=9, dropout=0.1, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        layers = []
        input_size = 2 * num_vector_feats * 59 + 3 * num_vector_feats + num_scalar_feats
        hidden_sizes = [input_size] + list(hidden_sizes)
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

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
