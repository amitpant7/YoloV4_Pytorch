import torch
from torch import nn
from .model_utils import CBLBlock, FeatureOut, CLBBlockx2, Concat, SpatialPyramidPooling


# ref: https://miro.medium.com/v2/resize:fit:1400/0*AZGFRB6kv9qmwYca.png

NECK_CONFIG = [
    # first half of the model
    [
        {
            "type": "cbl",
            "num_repeats": 2,
            "description": "A 3x3 and 1x1, actually 3 blocks but one already in backbone as 'p'.",
        },
        {"type": "spp", "description": "Spatial Pyramid Pooling"},
        {"type": "cbl", "num_repeats": 3, "description": "A 1x1, 3x3, 1x1"},
        {"type": "p", "description": "Take first output"},
        {
            "type": "U",
            "description": "CBL (channel reduce by half) + upsample",
        },  # after every upsample concatination with features from backbone
        {
            "type": "cbl",
            "num_repeats": 5,
            "out": 256,
            "description": "5 CBL layers with k: 1, 3, 1, 3",
        },
        {"type": "p", "description": "Take second output"},
        {"type": "U", "description": "CBL (reduce by half) + upsample"},
        {"type": "p", "description": "Take third output"},
    ],
    # second half of the model
    [
        {"type": "cbl", "out": 128, "num_repeats": 5},
        {"type": "p", "description": "Take first output"},
        {"type": "cbl_custom", "out": 256, "kernel_size": 3, "stride": 2},
        {"type": "C", "description": "Concat Layer"},
        {"type": "cbl", "out": 256, "num_repeats": 5},
        {"type": "p", "description": "Take first output"},
        {"type": "cbl_custom", "out": 512, "kernel_size": 3, "stride": 2},
        {"type": "C", "description": "Concat Layer"},
        {"type": "cbl", "out": 512, "num_repeats": 5},
        {"type": "p", "description": "Take first output"},
    ],
]


class PANnet(nn.Module):
    def __init__(self, cfg=NECK_CONFIG, in_channels=512):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels  # one 'p' block in backbone at the end has already reduced 1024->512
        self.first_part_layers = self._make_layers(cfg[0])
        self.second_part_layers = self._make_layers(cfg[1])

    def forward(self, in_features):  # list of three features from backbone
        x = in_features[2]  # as the network start processing from the last feature

        first_part_features = []  # store outputs of first part in order 512, 256, 256
        second_part_features = []

        count = 1  # to keep track of concatinations

        for layer in self.first_part_layers:
            if isinstance(layer, nn.Identity):
                first_part_features.append(x)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)
                x = torch.cat([x, in_features[count]], dim=1)
                count -= 1

            else:
                x = layer(x)

        # return first_part_features

        count = 1
        for layer in self.second_part_layers:
            if isinstance(layer, nn.Identity):
                second_part_features.append(x)

            elif isinstance(layer, Concat):
                x = layer(x, first_part_features[count])
                count -= 1

            else:
                x = layer(x)

        return second_part_features  # outputs from the neck

    def _make_layers(self, cfg):

        in_channels = self.in_channels
        all_layers = nn.ModuleList()

        for module in cfg:
            if module["type"] == "cbl":
                # will excute only in first part
                if module["num_repeats"] == 2:
                    all_layers += [CLBBlockx2(channels=in_channels)]
                    all_layers += [SpatialPyramidPooling()]

                    in_channels = (
                        4 * in_channels
                    )  # as SPP will increase in channels by 4

                elif module["num_repeats"] == 3:
                    # will excute only in first part
                    # Also reduce channels by 4 that were increased by SPP
                    all_layers += [
                        nn.Sequential(
                            CBLBlock(in_channels, in_channels // 4, 1),
                            CLBBlockx2(channels=in_channels // 4),
                        )
                    ]
                    in_channels = in_channels // 4

                elif module["num_repeats"] == 5:
                    out_channels = module["out"]
                    all_layers += [
                        nn.Sequential(
                            CBLBlock(in_channels, out_channels, 1),
                            CLBBlockx2(channels=out_channels),
                            CLBBlockx2(channels=out_channels),
                        )
                    ]
                    in_channels = out_channels

            elif module["type"] == "p":
                all_layers += [nn.Identity()]

            elif module["type"] == "U":
                all_layers += [
                    CBLBlock(in_channels, in_channels // 2, 1),
                    nn.Upsample(scale_factor=2),
                ]

                in_channels = in_channels
                # after every upsample concat will occur so channels remain so thou halfed by CBLblock

            # this will excute only for second half
            elif module["type"] == "cbl_custom":
                out_channels, ks, s = (
                    module["out"],
                    module["kernel_size"],
                    module["stride"],
                )
                all_layers += [CBLBlock(in_channels, out_channels, ks, s, 1)]
                in_channels = out_channels

            elif module["type"] == "C":
                all_layers += [Concat()]
                in_channels = 2 * in_channels

        self.in_channels = in_channels  # in channels for next call
        return all_layers
