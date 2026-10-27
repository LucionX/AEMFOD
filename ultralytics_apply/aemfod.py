# AEMFOD modules extracted from the author's YOLOv5 implementation.
# Integration target: Ultralytics 8.3.0 (YOLOv8 and YOLO11).
# Retain the original repository's GPL-3.0 notices when redistributing;
# Ultralytics integration is subject to its AGPL-3.0 license.
"""Active AEFPN/AWCF/ALF components, with no dormant experimental modules.

Place this file at ultralytics/nn/modules/aemfod.py. Standard Conv is reused
from Ultralytics so its normal Conv/BN fusion and export paths remain available.
Fusion paths themselves are defined by the supplied model YAML files.
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ("AWCF2", "AWCF3", "AWCF4", "ALF1", "ALF2", "AEFPNBlock")


class _Upsample(nn.Module):
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv(c1, c2, 1), nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        )

    def forward(self, x):
        return self.upsample(x)


class _Downsample(nn.Module):
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        self.downsample = nn.Sequential(Conv(c1, c2, scale_factor, scale_factor, 0))

    def forward(self, x):
        return self.downsample(x)


class _HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


class _HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = _HardSigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """Original coordinate attention, retaining the original state-dict keys."""
    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        oup = inp if oup is None else oup
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = _HardSwish()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat((x_h, x_w), dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h


class WeightedConcat(nn.Module):
    """One learned scalar per input, normalized exactly as in the source code.

    There is deliberately no ReLU/softmax/nonnegativity constraint on w.
    epsilon is retained as 1e-4; no new stabilization rule is introduced.
    """
    def __init__(self, inputs, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(inputs, dtype=torch.float32))
        self.epsilon = 0.0001

    def forward(self, x):
        weight = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        return torch.cat([weight[i] * value for i, value in enumerate(x)], self.d)


def _validate_fusion(c2, level, channels, count):
    if len(channels) != count or not 0 <= level < count:
        raise ValueError("Fusion requires %d ordered input levels and a valid target level" % count)
    if channels[level] != c2:
        raise ValueError("Target channels %s must equal actual target-input channels %s" % (c2, channels[level]))


class AWCF2(nn.Module):
    """Two-scale AWCF; args after parser expansion: c2, level, channels."""
    def __init__(self, inter_dim, level, channel):
        super().__init__()
        _validate_fusion(inter_dim, level, channel, 2)
        self.inter_dim, self.level = inter_dim, level
        if level == 0:
            self.upsample = _Upsample(channel[1], channel[0])
        else:
            self.downsample = _Downsample(channel[0], channel[1])
        self.bincat2 = WeightedConcat(2)
        self.ca = CoordinateAttention(inter_dim * 2)
        self.conv_sample = Conv(inter_dim * 2, inter_dim, 3, 1)

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        else:
            input1 = self.downsample(input1)
        return self.conv_sample(self.ca(self.bincat2([input1, input2])))


class AWCF3(nn.Module):
    """Three-scale AWCF, without unused conv_3/cot_3 or attention variants."""
    def __init__(self, inter_dim, level, channel):
        super().__init__()
        _validate_fusion(inter_dim, level, channel, 3)
        self.inter_dim, self.level = inter_dim, level
        if level == 0:
            self.upsample4x = _Upsample(channel[2], channel[0], 4)
            self.upsample2x = _Upsample(channel[1], channel[0], 2)
        elif level == 1:
            self.upsample2x1 = _Upsample(channel[2], channel[1], 2)
            self.downsample2x1 = _Downsample(channel[0], channel[1], 2)
        else:
            self.downsample2x = _Downsample(channel[1], channel[2], 2)
            self.downsample4x = _Downsample(channel[0], channel[2], 4)
        self.bincat3 = WeightedConcat(3)
        self.ca3 = CoordinateAttention(inter_dim * 3)
        self.conv_sample = Conv(inter_dim * 3, inter_dim, 3, 1)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2, input3 = self.upsample2x(input2), self.upsample4x(input3)
        elif self.level == 1:
            input3, input1 = self.upsample2x1(input3), self.downsample2x1(input1)
        else:
            input1, input2 = self.downsample4x(input1), self.downsample2x(input2)
        return self.conv_sample(self.ca3(self.bincat3([input1, input2, input3])))


class AWCF4(nn.Module):
    """Four-scale AWCF, retaining the original full AEMFOD fusion pathway."""
    def __init__(self, inter_dim, level, channel):
        super().__init__()
        _validate_fusion(inter_dim, level, channel, 4)
        self.inter_dim, self.level = inter_dim, level
        if level == 0:
            self.upsample8x = _Upsample(channel[3], channel[0], 8)
            self.upsample4x = _Upsample(channel[2], channel[0], 4)
            self.upsample2x = _Upsample(channel[1], channel[0], 2)
        elif level == 1:
            self.upsample4x1 = _Upsample(channel[3], channel[1], 4)
            self.upsample2x1 = _Upsample(channel[2], channel[1], 2)
            self.downsample2x1 = _Downsample(channel[0], channel[1], 2)
        elif level == 2:
            self.upsample2x2 = _Upsample(channel[3], channel[2], 2)
            self.downsample2x = _Downsample(channel[1], channel[2], 2)
            self.downsample4x = _Downsample(channel[0], channel[2], 4)
        else:
            self.downsample2x = _Downsample(channel[2], channel[3], 2)
            self.downsample4x = _Downsample(channel[1], channel[3], 4)
            self.downsample8x = _Downsample(channel[0], channel[3], 8)
        self.bincat4 = WeightedConcat(4)
        self.cord4 = CoordinateAttention(inter_dim * 4)
        self.conv_sample = Conv(inter_dim * 4, inter_dim, 3, 1)

    def forward(self, x):
        input1, input2, input3, input4 = x
        if self.level == 0:
            input2, input3, input4 = self.upsample2x(input2), self.upsample4x(input3), self.upsample8x(input4)
        elif self.level == 1:
            input1, input3, input4 = self.downsample2x1(input1), self.upsample2x1(input3), self.upsample4x1(input4)
        elif self.level == 2:
            input1, input2, input4 = self.downsample4x(input1), self.downsample2x(input2), self.upsample2x2(input4)
        else:
            input1, input2, input3 = self.downsample8x(input1), self.downsample4x(input2), self.downsample2x(input3)
        return self.conv_sample(self.cord4(self.bincat4([input1, input2, input3, input4])))


class ALF1(nn.Module):
    """Adjacent levels [coarse, target, fine] -> target; source Bifusion_1."""
    def __init__(self, c2, channels):
        super().__init__()
        if len(channels) != 3 or channels[1] != c2:
            raise ValueError("ALF1 middle input channels must equal output channels")
        self.upsample = _Upsample(c2, c2)
        self.downsample = _Downsample(c2, c2)
        self.cv1, self.cv2 = Conv(channels[0], c2, 1), Conv(channels[2], c2, 1)
        self.cv3 = Conv(c2 * 3, c2, 1)

    def forward(self, x):
        input1, input2, input3 = x
        input1 = self.upsample(self.cv1(input1))
        input3 = self.downsample(self.cv2(input3))
        return self.cv3(torch.cat((input1, input2, input3), dim=1))


class ALF2(nn.Module):
    """Two coarse inputs + one fine input; source Bifusion_2.

    The upsampler is SHARED across the first two branches, as in the original.
    """
    def __init__(self, c2, channels):
        super().__init__()
        if len(channels) != 3 or channels[2] != c2:
            raise ValueError("ALF2 third input channels must equal output channels")
        self.upsample = _Upsample(c2, c2)
        self.cv1, self.cv2 = Conv(channels[0], c2, 1), Conv(channels[1], c2, 1)
        self.cv3 = Conv(c2 * 3, c2, 1)

    def forward(self, x):
        input1, input2, input3 = x
        input1 = self.upsample(self.cv1(input1))
        input2 = self.upsample(self.cv2(input2))
        return self.cv3(torch.cat((input1, input2, input3), dim=1))


class _BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight, gain=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.silu(out + x)


class AEFPNBlock(nn.Module):
    """Original BasicBlock_n, with explicit equal-channel residual validation."""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        if c1 != c2:
            raise ValueError("AEFPNBlock is an identity residual block: c1 must equal c2")
        self.m = nn.Sequential(*(_BasicBlock(c2) for _ in range(n)))

    def forward(self, x):
        return self.m(x)
