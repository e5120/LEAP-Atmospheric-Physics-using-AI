import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel


class SeqConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 4*out_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4*out_channels),
            nn.Conv1d(4*out_channels, 2*out_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2*out_channels),
            nn.Conv1d(2*out_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        out = self.layers(x)
        out = x + out + F.avg_pool1d(out, kernel_size=x.size(2))
        return out


class ChannelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(seq_len, seq_len*out_channels, kernel_size=in_channels)
        self.conv2 = nn.Conv1d(seq_len, seq_len*out_channels, kernel_size=out_channels)
        self.conv3 = nn.Conv1d(seq_len, seq_len*out_channels, kernel_size=out_channels)
        self.bn1 = nn.BatchNorm1d(seq_len)
        self.bn2 = nn.BatchNorm1d(seq_len)
        self.bn3 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = F.relu(self.conv1(x)).reshape(-1, self.seq_len, self.out_channels)
        out = self.bn1(out)
        out = F.relu(self.conv2(out)).reshape(-1, self.seq_len, self.out_channels)
        out = self.bn2(out)
        out = F.relu(self.conv3(out)).reshape(-1, self.seq_len, self.out_channels)
        out = self.bn3(out)
        x = x + out
        x = x.permute(0, 2, 1)
        return x


class MixCNNModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=64, kernel_size=3,
                 num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(num_feats, out_channels, 1, padding="same")
        self.ch_cnn1 = ChannelConv1d(out_channels, out_channels, 60)
        self.seq_cnn1 = SeqConv1d(out_channels, out_channels, kernel_size)
        self.ch_bn1 = nn.BatchNorm1d(out_channels)
        self.seq_bn1 = nn.BatchNorm1d(out_channels)
        self.ch_cnn2 = ChannelConv1d(out_channels, out_channels, 60)
        self.seq_cnn2 = SeqConv1d(out_channels, out_channels, kernel_size)
        self.ch_bn2 = nn.BatchNorm1d(out_channels)
        self.seq_bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, 14, 1, padding="same")
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)

        out = self.conv(x)  # (bs, ch, seq_len)
        out = self.ch_cnn1(out)
        out = self.ch_bn1(out)
        out = self.seq_cnn1(out)
        out = self.seq_bn1(out)
        out = self.ch_cnn2(out)
        out = self.ch_bn2(out)
        out = self.seq_cnn2(out)
        out = self.seq_bn2(out)
        out = self.conv2(out)
        vector_out = out[:, :6].reshape(out.size(0), -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }
