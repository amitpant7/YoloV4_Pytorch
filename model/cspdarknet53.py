# Backbone implementation with pretrained weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


# mish(x) = x * tanh(log(1 + e^x))
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv2dBatchLeaky(nn.Module):
    """
    This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation="leaky",
        leaky_slope=0.1,
    ):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(k / 2) for k in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope
        # self.mish = Mish()

        # Layer
        if activation == "leaky":
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU(self.leaky_slope, inplace=True),
            )
        elif activation == "mish":
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                Mish(),
            )
        elif activation == "linear":
            self.layers = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=False,
                )
            )

    def __repr__(self):
        s = "{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class SmallBlock(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="mish"),
            Conv2dBatchLeaky(nchannels, nchannels, 3, 1, activation="mish"),
        )
        # conv_shortcut
        """
        参考 https://github.com/bubbliiiing/yolov4-pytorch
        shortcut后不接任何conv
        """
        # self.active_linear = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation='linear')
        # self.conv_shortcut = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation='mish')

    def forward(self, data):
        short_cut = data + self.features(data)
        # active_linear = self.conv_shortcut(short_cut)

        return short_cut


# Stage1  conv [256,256,3]->[256,256,32]


class Stage2(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        # stage2 32
        self.conv1 = Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, 2, activation="mish")
        self.split0 = Conv2dBatchLeaky(
            2 * nchannels, 2 * nchannels, 1, 1, activation="mish"
        )
        self.split1 = Conv2dBatchLeaky(
            2 * nchannels, 2 * nchannels, 1, 1, activation="mish"
        )

        self.conv2 = Conv2dBatchLeaky(2 * nchannels, nchannels, 1, 1, activation="mish")
        self.conv3 = Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, 1, activation="mish")

        self.conv4 = Conv2dBatchLeaky(
            2 * nchannels, 2 * nchannels, 1, 1, activation="mish"
        )

    def forward(self, data):
        conv1 = self.conv1(data)
        split0 = self.split0(conv1)
        split1 = self.split1(conv1)
        conv2 = self.conv2(split1)
        conv3 = self.conv3(conv2)

        shortcut = split1 + conv3
        conv4 = self.conv4(shortcut)

        route = torch.cat([split0, conv4], dim=1)
        return route


class Stage3(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        # stage3 128
        self.conv1 = Conv2dBatchLeaky(
            nchannels, int(nchannels / 2), 1, 1, activation="mish"
        )
        self.conv2 = Conv2dBatchLeaky(
            int(nchannels / 2), nchannels, 3, 2, activation="mish"
        )

        self.split0 = Conv2dBatchLeaky(
            nchannels, int(nchannels / 2), 1, 1, activation="mish"
        )
        self.split1 = Conv2dBatchLeaky(
            nchannels, int(nchannels / 2), 1, 1, activation="mish"
        )

        self.block1 = SmallBlock(int(nchannels / 2))
        self.block2 = SmallBlock(int(nchannels / 2))

        self.conv3 = Conv2dBatchLeaky(
            int(nchannels / 2), int(nchannels / 2), 1, 1, activation="mish"
        )

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)

        split0 = self.split0(conv2)
        split1 = self.split1(conv2)

        block1 = self.block1(split1)
        block2 = self.block2(block1)

        conv3 = self.conv3(block2)

        route = torch.cat([split0, conv3], dim=1)

        return route


# Stage4 Stage5 Stage6
class Stage(nn.Module):
    def __init__(self, nchannels, nblocks):
        super().__init__()
        # stage4 : 128
        # stage5 : 256
        # stage6 : 512
        self.conv1 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="mish")
        self.conv2 = Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, 2, activation="mish")
        self.split0 = Conv2dBatchLeaky(
            2 * nchannels, nchannels, 1, 1, activation="mish"
        )
        self.split1 = Conv2dBatchLeaky(
            2 * nchannels, nchannels, 1, 1, activation="mish"
        )
        blocks = []
        for i in range(nblocks):
            blocks.append(SmallBlock(nchannels))
        self.blocks = nn.Sequential(*blocks)
        self.conv4 = Conv2dBatchLeaky(nchannels, nchannels, 1, 1, activation="mish")

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)

        split0 = self.split0(conv2)
        split1 = self.split1(conv2)
        blocks = self.blocks(split1)
        conv4 = self.conv4(blocks)
        route = torch.cat([split0, conv4], dim=1)

        return route


class CsDarkNet53(nn.Module):
    def __init__(self):
        super(CsDarkNet53, self).__init__()

        input_channels = 32

        # Network
        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 1, activation="mish")
        self.stage2 = Stage2(input_channels)
        self.stage3 = Stage3(4 * input_channels)
        self.stage4 = Stage(4 * input_channels, 8)
        self.stage5 = Stage(8 * input_channels, 8)
        self.stage6 = Stage(16 * input_channels, 4)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)

        return [stage4, stage5, stage6]

    def _init_wts(model, path="cspdarknet53_backbone.pth"):
        wts = torch.load(path)
        model.load_state_dict(wts)
        print("Weight Copied")
        return model
