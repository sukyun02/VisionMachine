import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        channels = [16, 16*k, 32*k, 64*k]
        self.conv1    = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group1   = self._make_group(channels[0], channels[1], n, stride=1, dropout_rate=dropout_rate)
        self.group2   = self._make_group(channels[1], channels[2], n, stride=2, dropout_rate=dropout_rate)
        self.group3   = self._make_group(channels[2], channels[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn_final = nn.BatchNorm2d(channels[3])
        self.fc       = nn.Linear(channels[3], num_classes)
        self._init_weights()
    def _make_group(self, in_ch, out_ch, n_blocks, stride, dropout_rate):
        layers = [WideBasicBlock(in_ch, out_ch, stride=stride, dropout_rate=dropout_rate)]
        for _ in range(1, n_blocks):
            layers.append(WideBasicBlock(out_ch, out_ch, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn_final(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def wrn_28_10(num_classes=100, dropout_rate=0.3):
    return WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)

def wrn_40_10(num_classes=100, dropout_rate=0.3):
    return WideResNet(depth=40, widen_factor=10, dropout_rate=dropout_rate, num_classes=num_classes)
