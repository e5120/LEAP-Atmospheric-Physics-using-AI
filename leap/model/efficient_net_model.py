import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch import Tensor
from einops import rearrange
from timm.layers import DropPath

from leap.model import BaseModel


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    d_k = key.shape[-1] ** 0.5
    attn = torch.matmul(query, key.transpose(-1, -2)) / d_k
    attn = attn.softmax(dim=-1)
    return torch.matmul(attn, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, merge_type: str = "concat", bottleneck_ratio: int = 1, use_pos_emb: bool = False):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio
        self.num_heads = num_heads
        self.qkv = nn.Linear(input_dim, d_model * 3)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb_q = nn.Parameter(torch.randn(1, num_heads, 4, d_model // num_heads))
            self.pos_emb_k = nn.Parameter(torch.randn(1, num_heads, 4, d_model // num_heads))

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        if self.use_pos_emb:
            q = q + self.pos_emb_q
            k = k + self.pos_emb_k
        x = scaled_dot_product_attention(q, k, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class SharedQkMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, merge_type: str = "concat", bottleneck_ratio: int = 1, use_pos_emb: bool = True):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio

        self.num_heads = num_heads
        self.qv = nn.Linear(input_dim, d_model * 2)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, num_heads, 4, d_model // num_heads))

    def forward(self, x: Tensor) -> Tensor:
        qv = self.qv(x)
        qv = rearrange(qv, "b n (h d) -> b h n d", h=self.num_heads)
        q, v = torch.chunk(qv, 2, dim=-1)
        if self.use_pos_emb:
            q = q + self.pos_emb
        x = scaled_dot_product_attention(q, q, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class SharedQkvMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, merge_type: str = "concat", bottleneck_ratio: int = 1):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio
        self.num_heads = num_heads
        self.v = nn.Linear(input_dim, d_model)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type

    def forward(self, x: Tensor) -> Tensor:
        v = self.v(x)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        x = scaled_dot_product_attention(v, v, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class GeMPool1d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class SqueezeExcite(nn.Module):
    def __init__(self, hidden_dim: int, se_ratio: int, activation: type[nn.Module] = nn.ELU):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(inplace=True),
            nn.Conv2d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, ...] = (1, 1), stride: tuple[int, ...] = (1, 1),
                 padding: tuple[int, ...] | str = "same", groups: int = 1, activation: type[nn.Module] = nn.ELU, drop_path_rate: float = 0.0, skip: bool = False):
        super().__init__()
        assert (
            kernel_size >= stride
        ), f"kernel_size must be greater than stride. Got {kernel_size} and {stride}."

        match padding:
            case "valid":
                padding = (0, 0)
            case "same":
                padding = tuple([(k - s) // 2 for k, s in zip(kernel_size, stride)])
            case _:
                pass

        self.has_skip = skip and stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,  # type: ignore
                stride,  # type: ignore
                padding,  # type: ignore
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: tuple[int, ...], activation: type[nn.Module], se_ratio: int = 4,
                 skip: bool = True, drop_path_rate: float = 0.0, se_after_dw_conv: bool = False):
        super().__init__()
        self.has_skip = skip
        modules: list[nn.Module] = [
            ConvBnAct2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            )
        ]
        if se_after_dw_conv:
            modules.append(SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation))
        modules.append(ConvBnAct2d(hidden_dim, hidden_dim, activation=activation))
        if not se_after_dw_conv:
            modules.append(SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation))
        self.conv = nn.Sequential(*modules)
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, depth_multiplier: int, kernel_size: tuple[int, ...], activation: type[nn.Module],
                 se_ratio: int = 4, skip: bool = True, drop_path_rate: float = 0.0, se_after_dw_conv: bool = False):
        super().__init__()
        self.has_skip = skip
        modules: list[nn.Module] = [
            ConvBnAct2d(in_channels, hidden_dim, activation=activation),
            ConvBnAct2d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            ),
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(
                    hidden_dim * depth_multiplier,
                    se_ratio=se_ratio,
                    activation=activation,
                )
            )
        modules.append(
            ConvBnAct2d(
                hidden_dim * depth_multiplier,
                hidden_dim,
                activation=activation,
            ),
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            )
        self.inv_res = nn.Sequential(*modules)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        if self.has_skip and in_channels != hidden_dim:
            self.skip_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            if self.in_channels != self.hidden_dim:
                shortcut = self.skip_conv(shortcut)

            x = self.drop_path(x) + shortcut
        return x


class ResBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layer: nn.Module, pool_size: int = 2, skip: bool = True):
        super().__init__()
        self.has_skip = skip
        if self.has_skip:
            self.pool = nn.Conv2d(in_channels, out_channels, kernel_size=(1, pool_size), stride=(1, pool_size))
        self.layer = nn.Sequential(
            layer,
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_skip:
            return self.layer(x)
        return self.pool(x) + self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dims)
        # self.mhsa = SharedQkMultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)
        self.mhsa = MultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.norm(x))
        return x


class TransformerChannelMixer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[-1]
        x = rearrange(x, "b c ch t -> b (t ch) c")
        x = self.transformer(x)
        x = rearrange(x, "b (t ch) c -> b c ch t", t=T)
        return x


class EfficientNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        depth_multiplier: int = 4,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        kernel_sizes: list[int] = [3, 3, 3, 5, 5],
        pool_sizes: list[int] = [2, 2, 2, 2, 2],
        layers: list[int] | int = [1, 2, 2, 3, 3],
        skip_in_block: bool = True,
        skip_in_layer: bool = True,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
        use_ds_conv: bool = True,
        se_after_dw_conv: bool = False,
        use_channel_mixer: bool = False,
        channel_mixer_kernel_size: int = 3,
        mixer_type: str = "sc",
        transformer_merge_type: str = "add",
    ):
        super().__init__()
        if isinstance(layers, int):
            layers = [layers] * len(kernel_sizes)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth_multiplier = depth_multiplier
        self.temporal_pool_sizes = pool_sizes
        self.temporal_layers = layers
        self.num_eeg_channels = in_channels // 2
        self.drop_path_rate = drop_path_rate

        self.stem_conv = ConvBnAct2d(
            in_channels,
            hidden_dim,
            kernel_size=(1, stem_kernel_size),
            activation=activation,
            drop_path_rate=drop_path_rate,
            stride=(1, stem_stride),
        )

        self.efficient_net = nn.Sequential(
            *[
                ResBlock2d(
                    in_channels=self.layer_num_to_channel(i),
                    out_channels=self.layer_num_to_channel(i + 1),
                    layer=nn.Sequential(
                        *[
                            DepthWiseSeparableConv(
                                hidden_dim=hidden_dim * 2 ** (i - 1),
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            if i == 0 and use_ds_conv
                            else InvertedResidual(
                                in_channels=self.layer_num_to_channel(i) if ii == 0 else self.layer_num_to_channel(i + 1),
                                hidden_dim=self.layer_num_to_channel(i + 1),
                                depth_multiplier=depth_multiplier,
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            for ii in range(nl)
                        ],
                        (
                            InvertedResidual(
                                hidden_dim=self.layer_num_to_channel(i + 1),
                                kernel_size=(channel_mixer_kernel_size, 1),
                                activation=activation,
                                depth_multiplier=depth_multiplier,
                                se_ratio=depth_multiplier,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            if mixer_type == "ir"
                            else (
                                DepthWiseSeparableConv(
                                    hidden_dim=self.layer_num_to_channel(i + 1),
                                    kernel_size=(channel_mixer_kernel_size, 1),
                                    activation=activation,
                                    se_ratio=depth_multiplier,
                                    se_after_dw_conv=se_after_dw_conv,
                                )
                                if mixer_type == "sc"
                                else TransformerChannelMixer(
                                    self.layer_num_to_channel(i + 1),
                                    num_heads=depth_multiplier,
                                    merge_type=transformer_merge_type,
                                )
                            )
                        )
                        if use_channel_mixer
                        else nn.Identity(),
                    ),
                    pool_size=p,
                    skip=skip_in_block,
                )
                for i, (k, p, nl) in enumerate(zip(kernel_sizes, pool_sizes, layers))
            ]
        )

    def layer_num_to_channel(self, i: int) -> int:
        hidden_dim = self.hidden_dim
        if i % 2 == 0:
            return int(hidden_dim * 2 ** (i // 2))
        else:
            return int(hidden_dim * 2 ** ((i - 1) // 2) * 1.5)

    @property
    def out_channels(self) -> int:
        return self.layer_num_to_channel(len(self.temporal_layers))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_conv(x)
        x = self.efficient_net(x)  # b c ch t = 2, 64, 4, 16
        return x


class EfficientNetModel(BaseModel):
    def __init__(self, input_size, output_size, hidden_dim=64, stem_kernel_size=3, stem_stride=1, kernel_sizes=[3,3,3], pool_sizes=[2,2,2],
                 layers=3, depth_multiplier=4, head="gem", drop_rate=0.0, drop_path_rate=0.0, sim=False, skip_in_block=True, skip_in_layer=True,
                 use_ds_conv=False, se_after_dw_conv=False, use_channel_mixer=True, channel_mixer_kernel_size=3, mixer_type="sc", ignore_mask=None, delta=1.0):
        super().__init__(ignore_mask=ignore_mask, delta=delta)
        output_size = 368
        self.head = head
        self.sim = sim
        kwargs = {
            "in_channels": 1,
            "hidden_dim": hidden_dim,
            "depth_multiplier": depth_multiplier,
            "stem_kernel_size": stem_kernel_size,
            "stem_stride": stem_stride,
            "kernel_sizes": kernel_sizes,
            "pool_sizes": pool_sizes,
            "layers": layers,
            "skip_in_block": skip_in_block,
            "skip_in_layer": skip_in_layer,
            "drop_path_rate": drop_path_rate,
            "use_ds_conv": use_ds_conv,
            "se_after_dw_conv": se_after_dw_conv,
            "use_channel_mixer": use_channel_mixer,
            "channel_mixer_kernel_size": channel_mixer_kernel_size,
            "mixer_type": mixer_type,
        }
        self.backbone = EfficientNet1d(**kwargs)
        d_model = self.backbone.out_channels + 360 if sim else self.backbone.out_channels
        if head == "trans":
            self.transformer = Transformer(d_model=d_model, dim_feedforward=d_model * 4, batch_first=True,
                                           norm_first=True, custom_decoder=nn.Identity(), num_encoder_layers=2,
                                           nhead=4)
            self.pos_emb = nn.Parameter(torch.randn(1, 1, 25, 1))
        elif head == "flatten":
            self.fc = nn.Sequential(
                nn.Linear(d_model * 4 * 3, d_model * 4),
                nn.BatchNorm1d(d_model * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(d_model * 4, output_size))
        elif head == "none":
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(d_model, output_size))
        elif head == "gem":
            self.gem = GeMPool1d()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(d_model, output_size))
        else:
            raise ValueError(f"unknown head {head}")

    @staticmethod
    def sim(x):
        # 2, 64, 4, 60
        b, dim, pos, time = x.shape
        x1 = x.reshape(b, dim, pos, 1, time)
        x2 = x.reshape(b, dim, 1, pos, time)
        sim = F.cosine_similarity(x1, x2, dim=1)  # 2, 4, 4, 62
        rows, cols = torch.triu_indices(pos, pos, 1)
        sim = sim[:, rows, cols]  # 2, 6, 60
        return sim.flatten(1, 2)  # 2, 360

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, num_features, seq_len)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, num_features, seq_len)
        x = torch.cat([v, s], dim=1)
        x = x.unsqueeze(1)  # (bs, 1, num_features, seq_len)
        x = self.backbone(x)

        if self.head == "trans":
            b, dim, pos, time = x.shape
            x = x + self.pos_emb  # b, dim, pos, time
            x = x.permute(0, 2, 3, 1)  # b, pos, time, dim
            x = x.reshape(b, pos * time, dim)
            x = self.transformer.encoder(x)
            x = x.mean(dim=1)
        elif self.head == "flatten":
            x = x.flatten(1, 2)  # 2, 64 * 4, 62
            x = F.adaptive_avg_pool1d(x, 3)
            x = x.flatten(1, 2)
        elif self.head == "none":
            x = x.mean(dim=2).mean(dim=2)  # 2, 64, 4, 62
        elif self.head == "gem":
            if self.sim:
                sim = self.sim(x)
            x = x.flatten(2, 3)  # 2, 64, 4 * 62
            x = self.gem(x)  # b c t
            x = x.squeeze(2)
            if self.sim:
                x = torch.cat([x, sim], dim=1)

        logits = self.fc(x)
        return {
            "logits": logits,
        }
