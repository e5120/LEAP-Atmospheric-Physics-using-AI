import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from leap.model import BaseModel
from leap.model.modules import get_act_fn
from leap.utils import NUM_GRID


class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, stride=1, padding="same", bias=False)

    def forward(self, x):
        # (bs, ch, seq)
        out = F.adaptive_avg_pool1d(x, 1)
        out = out.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)
        out = F.sigmoid(out)
        return x * out


class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        assert out_channels % in_channels == 0
        self.pad = nn.ZeroPad1d((kernel_size-1, 0))
        self.dw_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        # (bs, ch, seq)
        out = self.pad(x)
        out = self.dw_conv(out)
        return out


class Conv1dBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, ch_kernel_size=5, expand=2, activation='swish', dropout=0.0, momentum=0.1):
        super().__init__()
        self.act_fn = get_act_fn(activation)

        expand_dim = expand * dim
        self.ffn1 = nn.Linear(dim, expand*dim)
        self.dw_conv = DepthWiseConv1d(expand_dim, expand_dim, kernel_size=kernel_size)
        self.conv_norm = nn.BatchNorm1d(expand_dim, momentum=momentum)
        self.eca = ECA(kernel_size=ch_kernel_size)
        self.ffn2 = nn.Linear(expand_dim, dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (bs, seq, ch)
        out = self.ffn1(x)
        out = self.act_fn(out)
        out = out.permute(0, 2, 1)
        out = self.dw_conv(out)
        out = self.conv_norm(out)
        out = self.eca(out)
        out = out.permute(0, 2, 1)
        out = self.ffn2(out)
        # out = self.act_fn(out)
        out = self.dropout(out)
        out = out + x
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60, dropout=0.0, pos_type="sinusoid"):
        super().__init__()
        assert pos_type in ["embedding", "sinusoid"]
        self.pos_type = pos_type
        if self.pos_type == "embedding":
            self.positional_embedding = nn.Embedding(max_len, d_model)
        else:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (bs, seq, ch)
        if self.pos_type == "embedding":
            pos_emb = self.positional_embedding(torch.arange(x.size(1)).to(x.device))
            x = x + pos_emb.unsqueeze(dim=0)
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(query, key, value, dropout=0.0):
    # (bs, seq, dim)
    d_k = key.shape[-1] ** 0.5
    attn = torch.matmul(query, key.transpose(-1, -2)) / d_k
    attn = attn.softmax(dim=-1)
    attn = F.dropout(attn, p=dropout)
    return torch.matmul(attn, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, merge_type="concat", bottleneck_ratio=1, bias=False, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        input_dim = d_model
        d_model = d_model // bottleneck_ratio
        self.num_heads = num_heads
        self.qkv = nn.Linear(input_dim, 3 * d_model, bias=bias)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim,
            bias=bias,
        )
        self.merge_type = merge_type
        self.dropout = dropout

    def forward(self, x):
        # (bs, seq, ch)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        x = scaled_dot_product_attention(q, k, v, dropout=self.dropout)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dims, num_heads, expand=4, attn_dropout=0.2, ffn_dropout=0.2, activation="swish", bias=False, eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims, eps=eps)
        self.mhsa = MultiHeadSelfAttention(dims, num_heads, bias=bias, dropout=attn_dropout)
        self.dropout = ffn_dropout
        self.norm2 = nn.LayerNorm(dims, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dims, expand*dims, bias=bias),
            get_act_fn(activation),
            nn.Linear(expand*dims, dims, bias=bias),
        )

    def forward(self, x):
        # (bs, seq, ch)
        out = self.norm1(x)
        out = self.mhsa(out)
        out = F.dropout(out, p=self.dropout)
        # out = F.dropout1d(out, p=self.dropout)
        x = x + out

        out = self.norm2(x)
        out = self.ffn(out)
        out = F.dropout(out, p=self.dropout)
        # out = F.dropout1d(out, p=self.dropout)
        x = x + out
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, dims, kernel_sizes=[5, 3, 1], ch_kernel_size=5, num_attn_layers=1, num_heads=4, conv_expand=2, attn_expand=2,
                 conv_dropout=0.2, attn_dropout=0.2, ffn_dropout=0.2, activation="swish", use_pos_emb=False, pos_type="sinusoid",
                 momentum=0.1, eps=1e-5, bias=False):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[
            Conv1dBlock(
                dims,
                kernel_size=kernel_size,
                ch_kernel_size=ch_kernel_size,
                expand=conv_expand,
                activation=activation,
                dropout=conv_dropout,
                momentum=momentum,
            )
            for kernel_size in kernel_sizes
        ])
        self.trans_blocks = nn.Sequential(*[
            TransformerBlock(
                dims, num_heads,
                expand=attn_expand,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                activation=activation,
                bias=bias,
                eps=eps,
            )
        ])
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_encoding = PositionalEncoding(dims, pos_type=pos_type)

    def forward(self, x):
        # (bs, seq, ch)
        x = self.conv_blocks(x)
        if self.use_pos_emb:
            x = self.pos_encoding(x)
        x = self.trans_blocks(x)
        return x


class ASLFRModelV2(BaseModel):
    def __init__(self, input_size, output_size, dims, num_layers=5, kernel_sizes=[5, 3, 1],
                 ch_kernel_size=5, num_attn_layers=1, num_heads=4,
                 conv_expand=2, attn_expand=2, conv_dropout=0.2, attn_dropout=0.2, ffn_dropout=0.2, head_dropout=0.4, activation="swish",
                 use_pos_emb=False, pos_type="sinusoid", momentum=0.1, eps=1e-5, bias=False, ignore_mask=None, use_aux=False, aux_weight=0.1, delta=1.0):
        super().__init__(ignore_mask=ignore_mask, use_aux=use_aux, aux_weight=aux_weight, delta=delta)
        in_channels, out_channels = 25, 14
        self.stem_conv1 = nn.Conv1d(in_channels, dims, 1, bias=bias)
        self.bn1 = nn.BatchNorm1d(dims, momentum=momentum)
        self.stem_conv2 = nn.Conv1d(in_channels, dims, 1, bias=bias)
        self.bn2 = nn.BatchNorm1d(dims, momentum=momentum)

        blocks1 = []
        for i in range(num_layers):
            blocks1.append(
                ConvTransBlock(
                    dims,
                    kernel_sizes=kernel_sizes,
                    ch_kernel_size=ch_kernel_size,
                    num_attn_layers=num_attn_layers,
                    num_heads=num_heads,
                    conv_expand=conv_expand,
                    attn_expand=attn_expand,
                    conv_dropout=conv_dropout,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    activation=activation,
                    use_pos_emb=use_pos_emb and i == 0,
                    pos_type=pos_type,
                    momentum=momentum,
                    eps=eps,
                    bias=bias,
                )
            )
        self.blocks1 = nn.Sequential(*blocks1)
        blocks2 = []
        for i in range(num_layers):
            blocks2.append(
                ConvTransBlock(
                    dims,
                    kernel_sizes=kernel_sizes,
                    ch_kernel_size=ch_kernel_size,
                    num_attn_layers=num_attn_layers,
                    num_heads=num_heads,
                    conv_expand=conv_expand,
                    attn_expand=attn_expand,
                    conv_dropout=conv_dropout,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    activation=activation,
                    use_pos_emb=use_pos_emb and i == 0,
                    pos_type=pos_type,
                    momentum=momentum,
                    eps=eps,
                    bias=bias,
                )
            )
        self.blocks2 = nn.Sequential(*blocks1)

        self.head1 = nn.Sequential(
            nn.Linear(dims, 2*dims),
            get_act_fn(activation),
            nn.Dropout(p=head_dropout),
            nn.Linear(2*dims, out_channels),
        )
        self.head2 = nn.Sequential(
            nn.Linear(dims, 2*dims),
            get_act_fn(activation),
            nn.Dropout(p=head_dropout),
            nn.Linear(2*dims, out_channels),
        )
        if self.use_aux:
            self.aux_decoder = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(dims, NUM_GRID),
            )


    def forward(self, batch):
        v = batch["x_vector"]  # (bs, 9, 60)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, 16, 60)
        x = torch.cat([v, s], dim=1)  # (bs, 25, 60)

        out1 = self.stem_conv1(x[:, :, :20])
        out1 = self.bn1(out1)
        out1 = out1.permute(0, 2, 1)
        out1 = self.blocks1(out1)
        # hidden_state = out.permute(0, 2, 1)
        out1 = self.head1(out1)
        out1 = out1.permute(0, 2, 1)

        out2 = self.stem_conv2(x[:, :, 20:])
        out2 = self.bn2(out2)
        out2 = out2.permute(0, 2, 1)
        out2 = self.blocks2(out2)
        # hidden_state = out.permute(0, 2, 1)
        out2 = self.head2(out2)
        out2 = out2.permute(0, 2, 1)

        out = torch.cat([out1, out2], dim=2)
        vector_out = out[:, :6].reshape(out.size(0), -1)
        scalar_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([scalar_out, vector_out], dim=1)
        return {
            "logits": logits,
            # "hidden_state": hidden_state,
        }
