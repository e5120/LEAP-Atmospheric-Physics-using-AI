import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import ResnetBlock


class CNNModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=128, resnet_blocks=9,
                 kernel_size=3, num_scalar_feats=16, num_vector_feats=9,
                 dropout=0.1, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(num_feats, out_channels, 1, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)
        self.resnet_block = self._make_resnet_layer(
            out_channels, kernel_size, blocks=resnet_blocks, dilation=1, p=dropout,
        )
        self.conv2 = nn.Conv1d(out_channels, 14, 1, padding="same")

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
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)

        out = self.conv(x)  # (bs, ch, seq_len)
        out = self.bn(F.silu(out))
        out = self.resnet_block(out)
        out = self.conv2(out)
        flat_out = out[:, :-6]
        seq_out = out[:, -6:]
        logits = torch.cat(
            [
                flat_out.mean(dim=-1),
                seq_out.reshape(seq_out.shape[0], -1),
            ],
            dim=1,
        )
        return {
            "logits": logits,
        }
