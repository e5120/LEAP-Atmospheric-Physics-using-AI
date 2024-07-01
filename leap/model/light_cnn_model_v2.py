import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel


class Conv1dBlock(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(num_channels, 4*num_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4*num_channels),
            nn.Conv1d(4*num_channels, 2*num_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2*num_channels),
            nn.Conv1d(2*num_channels, num_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_channels),
        )
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        out = self.layers(x)
        # out = x + out + out.mean(dim=1, keepdim=True) + out.mean(dim=2, keepdim=True)
        out = out + x + F.avg_pool1d(out, kernel_size=x.size(2))
        out = self.bn(out)
        return out


class LightCNNModelV2(BaseModel):
    def __init__(self, input_size, output_size, out_channels=64, kernel_size=3, num_layers=3,
                 num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(num_feats, out_channels, 1, padding="same")
        self.conv_blocks = nn.ModuleList([
            Conv1dBlock(out_channels, kernel_size)
            for _ in range(num_layers)
        ])
        self.conv2 = nn.Conv1d(out_channels, 14, 1, padding="same")

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)

        out = self.conv(x)  # (bs, ch, seq_len)
        for layer in self.conv_blocks:
            out = layer(out)
        out = self.conv2(out)
        vector_out = out[:, :6].reshape(out.size(0), -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }
