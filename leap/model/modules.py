import torch
import torch.nn as nn
import torch.nn.functional as F


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
        )

    def forward(self, x):
        out = self.block(x)
        out = x + out
        return out
