import torch
import torch.nn as nn
import torch.nn.functional as F

from leap.model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1, kernel_size
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride,
            dilation=dilation,
            padding=kernel_size//2,
            bias=True,
        )

    def forward(self, x):
        x = self.bn(x)
        x = F.silu(x)
        out = self.conv(x)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_type="mul"):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_channels//8, in_channels, kernel_size=1, padding=0)
        assert se_type in ["sum", "mul"]
        self.se_type = se_type

    def forward(self, x):
        x_se = F.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = F.elu(x_se)
        x_se = self.conv2(x_se)
        x_se = F.sigmoid(x_se)
        if self.se_type == "sum":
            out = torch.add(x, x_se)
        else:
            out = torch.multiply(x, x_se)
        return out


class REBlock(nn.Module):
    def __init__(self, num_channels, kernel_size=3, dilation=1, se_type="mul"):
        super().__init__()
        self.conv_block1 = ConvBlock(num_channels, num_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv_block2 = ConvBlock(num_channels, num_channels, kernel_size=kernel_size, dilation=dilation)
        self.se_block = SEBlock(num_channels, num_channels, se_type=se_type)

    def forward(self, x):
        x_re = self.conv_block1(x)
        x_re = self.conv_block2(x_re)
        x_re = self.se_block(x_re)
        out = torch.add(x, x_re)
        return out


class UNetWithSEModel(BaseModel):
    def __init__(self, input_size, output_size, out_channels=64, kernel_size=3,
                 in_depth=2, out_depth=3, se_type="mul", upsample_mode="nearest",
                 num_scalar_feats=16, num_vector_feats=9, ignore_mask=None):
        super().__init__(ignore_mask=ignore_mask)
        self.padding = nn.ZeroPad1d((4, 0))
        self.avg_pool1 = nn.AvgPool1d(kernel_size, stride=2, padding=kernel_size//2)
        self.avg_pool2 = nn.AvgPool1d(kernel_size, stride=4, padding=kernel_size//2)
        self.avg_pool3 = nn.AvgPool1d(kernel_size, stride=8, padding=kernel_size//2)

        in_channels = num_scalar_feats + num_vector_feats
        self.layer1 = self.make_down_layer(in_channels, out_channels, kernel_size, 1, in_depth, se_type)
        self.layer2 = self.make_down_layer(out_channels, 2*out_channels, kernel_size, 2, in_depth, se_type)
        self.layer3 = self.make_down_layer(2*out_channels+in_channels, 3*out_channels, kernel_size, 2, in_depth, se_type)
        self.layer4 = self.make_down_layer(3*out_channels+in_channels, 4*out_channels, kernel_size, 2, in_depth, se_type)
        self.layer5 = self.make_down_layer(4*out_channels+in_channels, 5*out_channels, kernel_size, 2, in_depth, se_type)

        self.up_layer1 = self.make_down_layer(9*out_channels, 4*out_channels, kernel_size, 1, out_depth, se_type)
        self.up_layer2 = self.make_down_layer(7*out_channels, 3*out_channels, kernel_size, 1, out_depth, se_type)
        self.up_layer3 = self.make_down_layer(5*out_channels, 2*out_channels, kernel_size, 1, out_depth, se_type)
        self.up_layer4 = self.make_down_layer(3*out_channels, out_channels, kernel_size, 1, out_depth, se_type)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

        self.out_conv = nn.Conv1d(out_channels, 14, kernel_size, padding="same")

    def make_down_layer(self, in_channels, out_channels, kernel_size, stride, depth, se_type):
        block = []
        block.append(ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        for _ in range(depth):
            block.append(
                REBlock(out_channels, kernel_size, se_type=se_type)
            )
        return nn.Sequential(*block)

    def forward(self, batch):
        v = batch["x_vector"]  # (bs, 9, 60)
        s = batch["x_scalar"].unsqueeze(dim=-1).repeat(1, 1, v.size(-1))  # (bs, 16, 60)
        x = torch.cat([v, s], dim=1)  # (bs, 25, 60)
        x = self.padding(x)          # 60 -> 64
        pool_x1 = self.avg_pool1(x)  # 64 -> 32
        pool_x2 = self.avg_pool2(x)  # 64 -> 16
        pool_x3 = self.avg_pool3(x)  # 64 -> 8

        # Encoder
        out_0 = self.layer1(x)       # 64 -> 64

        out_1 = self.layer2(out_0)   # 64 -> 32
        x = torch.cat([out_1, pool_x1], dim=1)

        out_2 = self.layer3(x)       # 32 -> 16
        x = torch.cat([out_2, pool_x2], dim=1)

        out_3 = self.layer4(x)       # 16 -> 8
        x = torch.cat([out_3, pool_x3], dim=1)

        x = self.layer5(x)           # 8 -> 4

        # Decoder
        up = self.upsample(x)        # 4 -> 8
        up = torch.cat([up, out_3], dim=1)
        up = self.up_layer1(up)

        up = self.upsample(up)       # 8 -> 16
        up = torch.cat([up, out_2], dim=1)
        up = self.up_layer2(up)

        up = self.upsample(up)       # 16 -> 32
        up = torch.cat([up, out_1], dim=1)
        up = self.up_layer3(up)

        up = self.upsample(up)       # 32 -> 64
        up = torch.cat([up, out_0], dim=1)
        up = self.up_layer4(up)

        out = self.out_conv(up[:, :, 4:])
        v_out = out[:, :6].reshape(out.size(0), -1)
        s_out = F.avg_pool1d(out[:, 6:], kernel_size=out.size(2)).squeeze(-1)
        logits = torch.cat([s_out, v_out], dim=1)
        return {
            "logits": logits,
        }
