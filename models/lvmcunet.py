import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from package.dysample import *
from mamba_ssm import Mamba
import bsconv.pytorch

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x


class CSF_Block(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.ca_h = ChannelAttention(dim_xh)
        self.ca_l = ChannelAttention(dim_xl)
        self.sa = SpatialAttention()
        self.conv_adjust = nn.Conv2d(dim_xh, dim_xl, 1)
        self.conv = bsconv.pytorch.BSConvU(dim_xl, dim_xl, kernel_size=3, stride=1, padding=1)

    def forward(self, x_h, x_l):
        fh = self.sa(self.ca_h(x_h))
        fl = self.sa(self.ca_l(x_l))

        fh = self.conv_adjust(fh)
        fh = F.interpolate(fh, size=[fl.size(2), fl.size(3)], mode='bilinear', align_corners=True)
        x = self.conv(fl) * self.conv(fh)
        return x


def rotate_every_two(x):
    x1 = x[:, ::2, :]
    x2 = x[:, 1::2, :]
    x = torch.stack((-x2, x1), dim=1)
    return x.flatten(1, 2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RelPos(nn.Module):
    def __init__(self, input_num, num_heads,):
        super().__init__()
        angle = 1.0 / (1000 ** torch.linspace(0, 1, input_num // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def forward(self, slen):
        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])
        retention_rel_pos = [sin, cos]

        return retention_rel_pos

class RVM_Block(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, num_heads=4):
        super().__init__()
        assert input_dim % 4 == 0, "input_dim must be divisible by 4"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.rel_pos = RelPos(input_dim, num_heads, )

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        sin, cos = self.rel_pos(n_tokens)  # 生成相对位置编码

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)
        x_mamba1 = theta_shift(self.mamba(x1), sin, cos) + self.skip_scale * x1
        x_mamba2 = theta_shift(self.mamba(x2), sin, cos) + self.skip_scale * x2
        x_mamba3 = theta_shift(self.mamba(x3), sin, cos) + self.skip_scale * x3
        x_mamba4 = theta_shift(self.mamba(x4), sin, cos) + self.skip_scale * x4

        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) + self.skip_scale * x_flat

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class LVMCUNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            DepthWiseConv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            DepthWiseConv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            RVM_Block(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            RVM_Block(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            RVM_Block(c_list[4], c_list[5]),
        )

        if bridge:
            self.CSF1 = CSF_Block(c_list[1], c_list[0])
            self.CSF2 = CSF_Block(c_list[2], c_list[1])
            self.CSF3 = CSF_Block(c_list[3], c_list[2])
            self.CSF4 = CSF_Block(c_list[4], c_list[3])
            self.CSF5 = CSF_Block(c_list[5], c_list[4])
            print("CSF_Block Module was used")

        self.decoder1 = nn.Sequential(
            RVM_Block(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            RVM_Block(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            RVM_Block(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            DepthWiseConv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            DepthWiseConv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0], num_classes, kernel_size=1),
        )

        self.BI1 = DySample(c_list[3], groups=4)
        self.BI2 = DySample(c_list[2], groups=4)
        self.BI3 = DySample(c_list[1], groups=4)
        self.BI4 = DySample(c_list[0], groups=4)
        self.BI5 = DySample(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        t6 = out

        if self.bridge:
            t1 = self.CSF1(t2, t1)
            t2 = self.CSF2(t3, t2)
            t3 = self.CSF3(t4, t3)
            t4 = self.CSF4(t5, t4)
            t5 = self.CSF5(t6, t5)

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32

        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16

        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8

        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4

        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2

        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        return torch.sigmoid(out0)
