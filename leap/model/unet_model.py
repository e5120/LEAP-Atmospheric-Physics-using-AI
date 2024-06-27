import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import MLPBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding="same",
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv1d(
                out_channels, out_channels, kernel_size,
                padding="same",
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.max_pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=3):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.max_pool = nn.MaxPool1d(2)
        self.conv_block = ConvBlock(out_channels+skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x, y):
        x = self.conv_t(x)
        x = torch.cat([x, y], dim=1)
        out = self.conv_block(x)
        return out


class UNetModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=32, hidden_sizes=[512, 256, 512],
                 kernel_size=3, num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        # スカラ用
        self.mlp_layer = MLPBlock(num_scalar_feats+60*num_vector_feats, 8, hidden_sizes=hidden_sizes)
        # ベクトル用
        self.zero_pad = nn.ZeroPad1d(2)
        in_channels = num_scalar_feats + num_vector_feats
        self.encoder_block1 = EncoderBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.encoder_block2 = EncoderBlock(out_channels, 2*out_channels, kernel_size=kernel_size)
        self.encoder_block3 = EncoderBlock(2*out_channels, 4*out_channels, kernel_size=kernel_size)
        self.encoder_block4 = EncoderBlock(4*out_channels, 8*out_channels, kernel_size=kernel_size)

        self.bottleneck = ConvBlock(8*out_channels, 16*out_channels, kernel_size=kernel_size)

        self.decoder_block1 = DecoderBlock(16*out_channels, 8*out_channels, 8*out_channels, kernel_size=kernel_size)
        self.decoder_block2 = DecoderBlock(8*out_channels, 4*out_channels, 4*out_channels, kernel_size=kernel_size)
        self.decoder_block3 = DecoderBlock(4*out_channels, 2*out_channels, 2*out_channels, kernel_size=kernel_size)
        self.decoder_block4 = DecoderBlock(2*out_channels, out_channels, out_channels, kernel_size=kernel_size)

        self.conv = nn.Conv1d(out_channels, 6, kernel_size=1, padding="same")

    def forward(self, batch):
        bs = batch["x_scalar"].size(0)
        y = torch.cat([batch["x_scalar"], batch["x_vector"].reshape(bs, -1)], dim=1)
        out_s = self.mlp_layer(y)

        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, num_features, seq_len)

        x = torch.cat([v, s], dim=1)
        x = self.zero_pad(x)

        x0, p0 = self.encoder_block1(x)
        x1, p1 = self.encoder_block2(p0)
        x2, p2 = self.encoder_block3(p1)
        x3, p3 = self.encoder_block4(p2)
        b1 = self.bottleneck(p3)

        d3 = self.decoder_block1(b1, x3)
        d4 = self.decoder_block2(d3, x2)
        d5 = self.decoder_block3(d4, x1)
        d6 = self.decoder_block4(d5, x0)

        out_v = self.conv(d6)[:, :, 2:-2]
        logits = torch.cat([out_s, out_v.reshape(bs, -1)], dim=1)
        return {
            "logits": logits,
        }
