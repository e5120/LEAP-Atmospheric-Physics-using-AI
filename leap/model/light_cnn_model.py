import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel

class LightCNNModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=64, resnet_blocks=9,
                 kernel_size=3, num_scalar_feats=16, num_vector_feats=9, seq_len=60, dropout=0.1):
        super().__init__()
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(
            num_feats, out_channels, kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.cnn_block1 = self._make_cnn_layer(out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.cnn_block2 = self._make_cnn_layer(out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, 14, 1, padding="same")
        self.fc = nn.Linear(out_channels, output_size)

    def _make_cnn_layer(self, num_channels, kernel_size):
        return nn.Sequential(
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

    def forward(self, batch):
        v = batch["vector"]  # (bs, num_features, seq_len)
        s = batch["scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)

        out = self.conv(x)  # (bs, ch, seq_len)
        out1 = self.cnn_block1(out)
        out = out + out1 + F.avg_pool1d(out1, kernel_size=x.size(2))
        out = self.bn(out)
        out = out + self.cnn_block2(out)
        out = self.conv2(out)
        seq_out = out[:, :6]
        flat_out = out[:, 6:]
        logits = torch.cat([
            seq_out.reshape(seq_out.shape[0], -1),
            flat_out.mean(dim=-1),
        ], dim=1)
        return {
            "logits": logits,
        }
