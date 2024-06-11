import torch.nn as nn

from .cspdarknet53 import *
from .head import Head
from .neck import PANnet


class YoloV4(nn.Module):
    def __init__(self, num_classes, no_of_anchors=3, backbone_wts_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.no_of_anchors = no_of_anchors
        self.backbone = CsDarkNet53()

        self.neck = PANnet()
        self.head = Head(self.num_classes, self.no_of_anchors)

        if backbone_wts_path is not None:
            self.backbone = self.backbone._init_wts(backbone_wts_path)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)

        out = [x.permute(0, 2, 3, 1) for x in out]

        out = [
            x.view(
                x.size(0),
                x.size(1),
                x.size(2),
                self.no_of_anchors,
                (self.num_classes + 5),
            )
            for x in out
        ]

        return [out[2], out[1], out[0]]
