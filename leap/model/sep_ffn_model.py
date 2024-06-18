import torch
import torch.nn as nn

from leap.model import BaseModel
from leap.utils import IN_COLUMNS, OUT_SCALAR_COLUMNS, OUT_VECTOR_COLUMNS


class SeparateFFNModel(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        layer_dict = {}
        for col in OUT_SCALAR_COLUMNS:
            layer_dict[col] = self._make_mlp(input_size, hidden_sizes, 1, dropout=dropout)
        for col in OUT_VECTOR_COLUMNS:
            layer_dict[col] = self._make_mlp(input_size, hidden_sizes, 60, dropout=dropout)
        self.layer_dict = nn.ModuleDict(layer_dict)
        self.loss_fn = nn.MSELoss(reduction="none")

    def _make_mlp(self, input_size, hidden_sizes, output_size, dropout):
        layers = []
        hidden_sizes = [input_size] + list(hidden_sizes)
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, batch):
        inputs = []
        for col in IN_COLUMNS:
            inputs.append(batch[col].reshape(batch[col].size(0), -1))
        inputs = torch.cat(inputs, dim=1)
        logits = []
        logits_dict = {}
        for col in OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS:
            out = self.layer_dict[col](inputs)
            logits.append(out)
            logits_dict[col] = out
        logits = torch.cat(logits, dim=1)
        return {
            "logits": logits,
            "logits_dict": logits_dict,
        }

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = None
        for col in OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS:
            if loss is None:
                loss = self.loss_fn(output["logits_dict"][col], batch[col]).mean(dim=0).sum()
            else:
                loss += self.loss_fn(output["logits_dict"][col], batch[col]).mean(dim=0).sum()
        output["loss"] = loss / (len(OUT_SCALAR_COLUMNS) + 60 * len(OUT_VECTOR_COLUMNS))
        return output
