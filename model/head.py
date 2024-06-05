import torch
import torch.nn as nn

from .model_utils import CBLBlock


class Head(nn.Module):
    def __init__(self, num_classes, no_of_anchors):
        super().__init__()
        self.output_channels = (5 + num_classes) * no_of_anchors

        self.out1 = nn.Sequential(
            CBLBlock(128, 256, 3, 1, 1), CBLBlock(256, self.output_channels, 1)
        )

        self.out2 = nn.Sequential(
            CBLBlock(256, 512, 3, 1, 1), CBLBlock(512, self.output_channels, 1)
        )

        self.out3 = nn.Sequential(
            CBLBlock(512, 1024, 3, 1, 1), CBLBlock(1024, self.output_channels, 1)
        )

    def forward(self, x):  # x is list of three features

        out1 = self.out1(x[0])
        out2 = self.out2(x[1])
        out3 = self.out3(x[2])

        return out1, out2, out3
