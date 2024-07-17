import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel
from leap.model.modules import PositionalEncoding, SqueezeformerBlock


class SqueezeFormerEncoder(nn.Module):
    def __init__(self, input_size=25, dim=256, kernel_size=3, groups=4, num_heads=4, num_layers=12,
                 conv_expand=4, attn_expand=4, num_conv_block=1, num_attn_block=1,
                 emb_dropout=0.1, conv_dropout=0.1, attn_dropout=0.1, mlp_dropout=0.1, drop_path=0.1,
                 activation='swish', prenorm=False, use_positional_encoding=False):
        super().__init__()
        self.embedding = nn.Linear(input_size, dim)
        self.use_positional_encoding = use_positional_encoding
        if self.use_positional_encoding:
            self.pos_embedding = PositionalEncoding(dim, max_len=60, dropout=0.0, pos_type="embedding")
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            SqueezeformerBlock(
                dim=dim,
                kernel_size=kernel_size,
                groups=groups,
                num_heads=num_heads,
                conv_expand=conv_expand,
                attn_expand=attn_expand,
                num_conv_block=num_conv_block,
                num_attn_block=num_attn_block,
                conv_dropout=conv_dropout,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                drop_path=drop_path,
                activation=activation,
                prenorm=prenorm,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # (bs, 60, 25)
        x = self.embedding(x)
        if self.use_positional_encoding:
            x = self.pos_embedding(x)
        x = self.emb_dropout(x)
        x = self.norm(x)

        outputs = []
        for layer in self.layers:
            x = layer(x)
            # x = x + layer(x)
            outputs.append(x)
        return outputs


class SqueezeFormerModel(BaseModel):
    def __init__(self, input_size, output_size, dim=128, groups=None, num_layers=12,
                 num_heads=4, kernel_size=3, num_conv_block=1, num_attn_block=1,
                 conv_expand=4, attn_expand=4, prenorm=False, activation="swish", drop_path=0.1,
                 emb_dropout=0.1, conv_dropout=0.1, attn_dropout=0.1, mlp_dropout=0.1, dropout=0.1,
                 num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        input_size = num_scalar_feats + num_vector_feats
        groups = dim if groups is None else groups
        self.encoder = SqueezeFormerEncoder(
            input_size,
            dim=dim,
            kernel_size=kernel_size,
            groups=groups,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_expand=conv_expand,
            attn_expand=attn_expand,
            num_conv_block=num_conv_block,
            num_attn_block=num_attn_block,
            emb_dropout=emb_dropout,
            conv_dropout=conv_dropout,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            drop_path=drop_path,
            activation=activation,
            prenorm=prenorm,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(dim, 14),
        )

    def forward(self, batch):
        v = batch["x_vector"]                                             # (bs,  9, 60)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, 16, 60)
        x = torch.cat([v, s], dim=1)                                      # (bs, 25, 60)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)[-1]
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        v_out = x[:, :6].reshape(x.size(0), -1)
        s_out = F.avg_pool1d(x[:, 6:], kernel_size=x.size(2)).squeeze(-1)
        logits = torch.cat([s_out, v_out], dim=1)
        return {
            "logits": logits,
        }
