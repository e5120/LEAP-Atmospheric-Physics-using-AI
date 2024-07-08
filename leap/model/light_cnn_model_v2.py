import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import get_act_fn, PositionalEncoding
from leap.model.unet_with_se_model import SEBlock


class Conv1dBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, activation="relu", expand=4, num_se_layers=3):
        super().__init__()
        out_channels1 = num_channels * expand
        out_channels2 = num_channels * expand // 2
        self.layers = nn.Sequential(
            nn.Conv1d(num_channels, out_channels1, kernel_size, padding="same"),
            get_act_fn(activation),
            nn.BatchNorm1d(out_channels1),
            nn.Conv1d(out_channels1, out_channels2, kernel_size, padding="same"),
            get_act_fn(activation),
            nn.BatchNorm1d(out_channels2),
            nn.Conv1d(out_channels2, num_channels, kernel_size, padding="same"),
            get_act_fn(activation),
            nn.BatchNorm1d(num_channels),
        )
        self.se_layers = nn.Sequential(*[
            SEBlock(num_channels, num_channels, se_type="mul")
            for _ in range(num_se_layers)
        ])
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        out = self.layers(x)
        out = self.se_layers(out)
        # out = x + out + out.mean(dim=1, keepdim=True) + out.mean(dim=2, keepdim=True)
        out = out + x + F.avg_pool1d(out, kernel_size=x.size(2))
        out = self.bn(out)
        return out


class LightCNNModelV2(BaseModel):
    def __init__(self, input_size, output_size, out_channels=64, kernel_size=3, num_layers=3,
                 activation="relu", use_positional_embedding=False, conv_expand=4, num_se_layers=3,
                 num_scalar_feats=16, num_vector_feats=9,
                 ignore_mask=None, use_aux=False, aux_weight=0.1):
        super().__init__(ignore_mask=ignore_mask, use_aux=use_aux, aux_weight=aux_weight)
        num_feats = num_scalar_feats + num_vector_feats
        self.conv = nn.Conv1d(num_feats, out_channels, 1, padding="same")
        self.use_pos_emb = use_positional_embedding
        if self.use_pos_emb:
            self.pos_emb = PositionalEncoding(out_channels, max_len=60, pos_type="sinusoid")
        self.conv_blocks = nn.ModuleList([
            Conv1dBlock(
                out_channels, kernel_size,
                activation=activation,
                expand=conv_expand,
                num_se_layers=num_se_layers,
            )
            for _ in range(num_layers)
        ])
        self.conv2 = nn.Conv1d(out_channels, 14, 1, padding="same")

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)
        out = self.conv(x)  # (bs, ch, seq_len)
        if self.use_pos_emb:
            out = out.permute(0, 2, 1)
            out = self.pos_emb(out)
            out = out.permute(0, 2, 1)
        for layer in self.conv_blocks:
            out = layer(out)
        out = self.conv2(out)
        vector_out = out[:, :6].reshape(out.size(0), -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
        }
