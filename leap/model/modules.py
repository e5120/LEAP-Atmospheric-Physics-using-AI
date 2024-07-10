import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_act_fn(activation):
    if activation == 'swish':
        return nn.SiLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'mish':
        return nn.Mish()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError


class GLU(nn.Module):
    def __init__(self, dim: int, activation: str = 'sigmoid') -> None:
        super(GLU, self).__init__()
        self.dim = dim
        self.activation = get_act_fn(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(Mlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.act = get_act_fn(activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)
        return x


class GLUMlp(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        expand: int = 4,
        dropout: float = 0.1,
        bias : bool = True,
        activation: str = 'gelu'
    ) -> None:
        super(GLUMlp, self).__init__()

        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)
        # self.do2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        # x = self.do2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ScaleBiasLayer(nn.Module):
    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(1, 1, d_model))
            self.bias = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_buffer('scale', torch.ones(1, 1, d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(1, 1, d_model), persistent=True)

    def forward(self, x):
        return x * self.scale + self.bias


class Conv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, groups=4, stride=1,
                 conv_dropout=0.0, mlp_dropout=0.0, drop_path=0.0,
                 expand=4, activation='swish', prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.stride = stride

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.expand_conv = nn.Linear(dim, 2*dim)
        self.glu = GLU(dim=-1, activation=activation)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding="same", groups=groups)
        self.conv_norm = nn.BatchNorm1d(dim, momentum=0.05)
        # self.conv_norm = nn.BatchNorm1d(dim)
        self.conv_act = get_act_fn(activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        # self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def forward(self, inputs):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = self.conv_norm(x)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0,2,1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        # x = self.mlp_dropout(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class AltAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0):
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(dropout)

    def forward(self, inputs):
        qkv = self.qkv(inputs)
        qkv = qkv.view(-1, inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.dim // self.num_heads] * 3, dim=-1)

        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], self.dim)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class AltBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2,
                 mlp_dropout=0.2, drop_path=0., activation='gelu', prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = AltAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)
        # self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout)
        self.drop1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)

        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def forward(self, inputs):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.self_attn(x)
        # x, _ = self.self_attn(x, x, x)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x

        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class SqueezeformerBlock(nn.Module):
    def __init__(self, dim=256, kernel_size=3, groups=4, num_heads=4,
                 conv_expand=4, attn_expand=4, num_conv_block=1, num_attn_block=1,
                 conv_dropout=0.1, attn_dropout=0.1, mlp_dropout=0.1, drop_path=0.1,
                 activation='swish', prenorm=True):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(
                dim=dim,
                kernel_size=kernel_size,
                groups=groups,
                stride=1,
                conv_dropout=conv_dropout,
                mlp_dropout=mlp_dropout,
                drop_path=drop_path,
                expand=conv_expand,
                activation=activation,
                prenorm=prenorm,
            )
            for _ in range(num_conv_block)
        ])
        self.attn_blocks = nn.ModuleList([
            AltBlock(
                dim=dim,
                num_heads=num_heads,
                expand=attn_expand,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                drop_path=drop_path,
                activation=activation,
                prenorm=prenorm,
            )
            for _ in range(num_attn_block)
        ])

    def forward(self, x):
        # 下の順番どっちがいいのかわからない
        for attn_block in self.attn_blocks:
            x = attn_block(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[], p=0.0, activation="swish"):
        super().__init__()
        mlp_layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            mlp_layers += [
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.LayerNorm(hidden_sizes[i+1]),
                get_act_fn(activation),
                nn.Dropout(p=p),
            ]
        mlp_layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp_layers(x)


class ResnetBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, dilation, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 1, padding="same"),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_channels),
            nn.Conv1d(
                num_channels, num_channels, kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                # bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_channels),
            nn.Conv1d(num_channels, num_channels, 1, padding="same"),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(num_channels),
        )

    def forward(self, x):
        out = self.block(x)
        out = x + out + F.avg_pool1d(out, kernel_size=out.size(2))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60, dropout=0.1, pos_type="embedding"):
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
        if self.pos_type == "embedding":
            pos_emb = self.positional_embedding(torch.arange(x.size(1)).to(x.device))
            x = x + pos_emb.unsqueeze(dim=0)
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
