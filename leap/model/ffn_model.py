import torch.nn as nn

from leap.model import BaseModel


class FFNModel(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1):
        super().__init__()
        layers = []
        hidden_sizes = [input_size] + list(hidden_sizes)
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        logits = self.layers(batch["input"].float())
        return {
            "logits": logits,
        }
