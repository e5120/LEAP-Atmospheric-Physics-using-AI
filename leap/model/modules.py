import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[], p=0.0):
        super().__init__()
        mlp_layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            mlp_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            mlp_layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            mlp_layers.append(nn.LeakyReLU(inplace=True))
            mlp_layers.append(nn.Dropout(p=p))
        mlp_layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp_layers(x)


class ResnetBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, dilation, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p, inplace=True),
            nn.Conv1d(
                num_channels, num_channels, kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_channels, num_channels, kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.Dropout(p=p, inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        out = x + out
        return out
