import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import ResnetBlock


class CNNModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=128, resnet_blocks=9,
                 kernel_size=3, num_scalar_feats=16, num_vector_feats=9, seq_len=60, dropout=0.1):
        super().__init__()
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(
            num_feats, out_channels, kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=False,
        )
        self.resnet_block = self._make_resnet_layer(
            out_channels, kernel_size, blocks=resnet_blocks, dilation=1, p=dropout,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        # self.avg_pools = nn.AvgPool1d(kernel_size=down_sampling, stride=down_sampling, padding=down_sampling//2-1)
        # resnet_features = out_channels * ((seq_len*len(kernel_sizes)//2 // 2**resnet_blocks -1) + 1)
        # self.fc = nn.Linear(out_channels * seq_len, output_size)
        self.fc = nn.Linear(out_channels, output_size)

    def _make_resnet_layer(self, num_channels, kernel_size, blocks=9, dilation=1, p=0.0):
        layers = []
        for _ in range(blocks):
            layers.append(
                ResnetBlock(
                    num_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    p=p,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, batch):
        v = batch["vector"]  # (bs, num_features, seq_len)
        s = batch["scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)

        out = self.conv(x)  # (bs, ch, seq_len)
        out = self.resnet_block(out)
        out = self.bn(out)
        out = F.relu(out)
        # out = self.avg_pools(out)
        out = F.avg_pool1d(out, kernel_size=x.size(2)).squeeze(dim=-1)
        out = out.reshape(out.shape[0], -1)
        logits = self.fc(out)

        return {
            "logits": logits,
        }
