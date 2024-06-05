# all utils used to build model and layers.
import torch
import torch.nn as nn


class CNNBlock(nn.Module):  ## CBM block
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))


class CBLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats

        for i in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, 1),
                    CNNBlock(channels // 2, channels, 3, padding=1),
                )
            ]

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class CSPBlock(nn.Module):
    def __init__(self, channels, repeat):
        super().__init__()
        self.conv_first = CNNBlock(
            channels, channels // 2, 1
        )  # to split in channels in two parts

        self.residual = ResidualBlock(channels // 2, num_repeats=repeat)
        self.conv_half = CNNBlock(
            channels // 2, channels // 2, 1
        )  # the 1*1 conv block at the end of non-shortcut block

        self.final_conv1 = CNNBlock(channels, channels, 1)

    def forward(self, x):
        # part 1, direct pass
        x1 = self.conv_first(x)

        # part2, secodn part
        x2 = self.conv_first(x)
        x2 = self.residual(x2)
        x2 = self.conv_half(x2)

        x = torch.cat([x1, x2], dim=1)
        out = self.final_conv1(x)

        return out


# Takes the pyramid features out, half the channels, contains CLB block ref:https://aiacademy.tw/yolo-v4-intro/
class FeatureOut(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cbl = CBLBlock(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x):
        return self.cbl(x)


## Extra Modules for neck

POOL_SIZE = (5, 9 ,13)
OUTPUT_SIZE = (13,13)

class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_sizes = POOL_SIZE
        self.output_size = OUTPUT_SIZE
        self.pool_layers = self._make_pool_layers()

    def forward(self, x):
        spp = [pool(x) for pool in self.pool_layers]
        x = [x]+spp
        
        return torch.cat(x, dim=1)
    
    
    def _make_pool_layers(self):
        pool_layers = []
        for pool_size in self.pool_sizes:
            maxpool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
            pool_layers.append(maxpool)
        return nn.ModuleList(pool_layers)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class CLBBlockx2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            CBLBlock(channels, channels, 3, 1, 1), CBLBlock(channels, channels, 1)
        )

    def forward(self, x):
        return self.layers(x)
