"""
PyramidNet-272 α200 with ShakeDrop Regularization
Reference: Han et al. (2017) "Deep Pyramidal Residual Networks"
           Yamada et al. (2019) "ShakeDrop Regularization for Deep Residual Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# ShakeDrop
# ---------------------------------------------------------------------------

class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training, p_drop, alpha_range):
        gate = x.new_empty(1).bernoulli_(1 - p_drop)
        ctx.save_for_backward(gate)
        ctx.training = training
        if training:
            alpha = x.new_empty(x.size(0)).uniform_(*alpha_range)
            alpha = alpha.view(-1, *([1] * (x.dim() - 1)))
            return (gate + alpha - gate * alpha) * x
        return gate * x

    @staticmethod
    def backward(ctx, grad_output):
        gate, = ctx.saved_tensors
        if ctx.training:
            beta = grad_output.new_empty(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(-1, *([1] * (grad_output.dim() - 1)))
            return (gate + beta - gate * beta) * grad_output, None, None, None
        return grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


# ---------------------------------------------------------------------------
# Basic Building Blocks
# ---------------------------------------------------------------------------

class BNReLUConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
        )


class PyramidBottleneckBlock(nn.Module):
    """Bottleneck residual block used in PyramidNet-272."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, p_shakedrop):
        super().__init__()
        bottleneck_ch = out_channels // self.expansion

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_ch, 1, bias=False)

        self.bn2   = nn.BatchNorm2d(bottleneck_ch)
        self.conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, 3,
                               stride=stride, padding=1, bias=False)

        self.bn3   = nn.BatchNorm2d(bottleneck_ch)
        self.conv3 = nn.Conv2d(bottleneck_ch, out_channels, 1, bias=False)

        self.bn4   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True) if stride > 1
            else nn.Identity(),
        )
        self.shortcut_pad = out_channels - in_channels  # zero-pad channels

        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        out = self.bn4(out)
        out = self.shake_drop(out)

        sc = self.shortcut(x)
        # zero-pad shortcut to match channel dimension
        if self.shortcut_pad > 0:
            sc = F.pad(sc, [0, 0, 0, 0, 0, self.shortcut_pad])

        return out + sc


# ---------------------------------------------------------------------------
# PyramidNet
# ---------------------------------------------------------------------------

class PyramidNet(nn.Module):
    """
    PyramidNet for CIFAR-100.

    Args:
        depth       : total network depth (e.g. 272)
        alpha       : channel-widening amount (e.g. 200)
        num_classes : output classes (100 for CIFAR-100)
        bottleneck  : use bottleneck blocks (True for depth >= 164)
        shakedrop   : enable ShakeDrop regularization
    """

    def __init__(self, depth=272, alpha=200, num_classes=100,
                 bottleneck=True, shakedrop=True):
        super().__init__()
        assert (depth - 2) % 9 == 0, "depth must be 9n+2 for bottleneck"
        n = (depth - 2) // 9           # blocks per stage
        block = PyramidBottleneckBlock

        # Channel schedule
        self.in_channels = 16
        step = alpha / (3 * n)         # additive widening per block

        # Layer 0: initial conv
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # Build stages
        self.layer1, _ = self._make_layer(block, n, step,  1, shakedrop, 0,     3*n)
        self.layer2, _ = self._make_layer(block, n, step,  2, shakedrop, n,     3*n)
        self.layer3, _ = self._make_layer(block, n, step,  1, shakedrop, 2*n,   3*n)

        self.bn_final = nn.BatchNorm2d(self.in_channels)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(self.in_channels, num_classes)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, step, stride,
                    shakedrop, block_idx_start, total_blocks):
        layers = []
        for i in range(num_blocks):
            out_ch = round(16 + step * (block_idx_start + i + 1))
            out_ch = out_ch * block.expansion
            p_drop = 0.5 * (block_idx_start + i) / (total_blocks - 1) if shakedrop else 0.0
            layers.append(block(self.in_channels, out_ch,
                                stride if i == 0 else 1, p_drop))
            self.in_channels = out_ch
        return nn.Sequential(*layers), None

    def forward(self, x):
        out = self.bn1(self.conv1(x))          # no ReLU yet (pre-activation style)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out), inplace=True)
        out = self.pool(out).flatten(1)
        return self.fc(out)


def pyramidnet272(num_classes=100, **kwargs):
    return PyramidNet(depth=272, alpha=200, num_classes=num_classes,
                      bottleneck=True, shakedrop=True, **kwargs)


if __name__ == "__main__":
    model = pyramidnet272()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {y.shape}")
    print(f"Total params : {total / 1e6:.2f} M")
