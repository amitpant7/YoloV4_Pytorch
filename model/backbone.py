import torch
import torch.nn as nn
from model.model_utils import *

# implementation of YOLOv4 backbone CSPDarknet53

# ref https://miro.medium.com/v2/resize:fit:1400/0*AZGFRB6kv9qmwYca.png
BACKBONE_CFG = [
    [32, 3, 1],
    [64, 3, 2],
    ("csp", 1),  # csp block, times repeat
    [128, 3, 2],
    ("csp", 2),
    [256, 3, 2],
    ("csp", 8),
    "p",  # this is where we take first feature out,
    [512, 3, 2],
    ("csp", 8),
    "p",
    [1024, 3, 2],
    ("csp", 4),  # second feature out
    "p",  # third feature out
]


class CSPDarknet53(nn.Module):
    def __init__(self, cfg=BACKBONE_CFG):
        super().__init__()
        self.cfg = cfg
        self.layers = self._make_layers(in_channels=3)

    def forward(self, x):
        features = []

        for layer in self.layers:

            if isinstance(layer, FeatureOut):
                features.append(layer(x))

            else:
                x = layer(x)

        return [features[0], features[1], features[2]]

    def _make_layers(self, in_channels):
        all_layers = nn.ModuleList()

        for module in self.cfg:
            if type(module) == list:
                out_channels, ks, s = module
                all_layers += [CNNBlock(in_channels, out_channels, ks, s, padding=1)]
                in_channels = out_channels

            elif type(module) == tuple:
                _, repeat = module
                all_layers += [CSPBlock(channels=in_channels, repeat=repeat)]

            elif type(module) == str:
                if module == "p":
                    all_layers += [FeatureOut(in_channels)]

        return all_layers
