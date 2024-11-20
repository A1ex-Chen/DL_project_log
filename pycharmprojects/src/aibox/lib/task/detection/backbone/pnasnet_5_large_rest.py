from typing import Tuple

import pretrainedmodels
from torch import nn, Tensor

from . import Backbone


class PNASNet_5_Large(Backbone):

    class Conv1(nn.Module):

        def __init__(self, pnasnet_5_large: nn.Module):
            super().__init__()
            self.conv_0 = pnasnet_5_large.conv_0
            self.x_conv_0 = None

        def forward(self, x: Tensor) -> Tensor:
            self.x_conv_0 = self.conv_0(x)
            return self.x_conv_0

    class Conv2(nn.Module):

        def __init__(self, pnasnet_5_large: nn.Module):
            super().__init__()
            self.cell_stem_0 = pnasnet_5_large.cell_stem_0

        def forward(self, x_conv_0: Tensor) -> Tensor:
            x_stem_0 = self.cell_stem_0(x_conv_0)
            return x_stem_0

    class Conv3(nn.Module):

        def __init__(self, pnasnet_5_large: nn.Module, conv1: 'PNASNet_5_Large.Conv1'):
            super().__init__()
            self.conv1 = conv1
            self.cell_stem_1 = pnasnet_5_large.cell_stem_1
            self.cell_0 = pnasnet_5_large.cell_0
            self.cell_1 = pnasnet_5_large.cell_1
            self.cell_2 = pnasnet_5_large.cell_2
            self.cell_3 = pnasnet_5_large.cell_3
            self.x_cell_2 = None

        def forward(self, x_stem_0: Tensor) -> Tensor:
            x_stem_1 = self.cell_stem_1(self.conv1.x_conv_0, x_stem_0)
            x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
            x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
            self.x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
            x_cell_3 = self.cell_3(x_cell_1, self.x_cell_2)
            return x_cell_3

    class Conv4(nn.Module):

        def __init__(self, pnasnet_5_large: nn.Module, conv3: 'PNASNet_5_Large.Conv3'):
            super().__init__()
            self.conv3 = conv3
            self.cell_4 = pnasnet_5_large.cell_4
            self.cell_5 = pnasnet_5_large.cell_5
            self.cell_6 = pnasnet_5_large.cell_6
            self.cell_7 = pnasnet_5_large.cell_7

        def forward(self, x_cell_3: Tensor) -> Tensor:
            x_cell_4 = self.cell_4(self.conv3.x_cell_2, x_cell_3)
            x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
            x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
            x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
            return x_cell_7

    class Conv5(nn.Module):

        def __init__(self, pnasnet_5_large: nn.Module):
            super().__init__()
            self.cell_8 = pnasnet_5_large.cell_8
            self.cell_9 = pnasnet_5_large.cell_9
            self.cell_10 = pnasnet_5_large.cell_10
            self.cell_11 = pnasnet_5_large.cell_11

        def forward(self, x_cell_7: Tensor) -> Tensor:
            x_cell_8 = self.cell_8(x_cell_7, x_cell_7)  # modify from `cell_8(x_cell_6, x_cell_7)` for compatibility
            x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
            x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
            x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
            return x_cell_11



    @staticmethod

    @staticmethod


    class Conv2(nn.Module):



    class Conv3(nn.Module):



    class Conv4(nn.Module):



    class Conv5(nn.Module):



    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        pnasnet_5_large = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet' if self.pretrained else None)

        # x = torch.randn(1, 3, 640, 480)
        # x_conv_0 = pnasnet_5_large.conv_0(x)                        # (1, 96, 319, 239)
        # x_stem_0 = pnasnet_5_large.cell_stem_0(x_conv_0)            # [1, 270, 160, 120)
        # x_stem_1 = pnasnet_5_large.cell_stem_1(x_conv_0, x_stem_0)  # (1, 540, 80, 60)
        # x_cell_0 = pnasnet_5_large.cell_0(x_stem_0, x_stem_1)       # (1, 1080, 80, 60)
        # x_cell_1 = pnasnet_5_large.cell_1(x_stem_1, x_cell_0)       # (1, 1080, 80, 60)
        # x_cell_2 = pnasnet_5_large.cell_2(x_cell_0, x_cell_1)       # (1, 1080, 80, 60)
        # x_cell_3 = pnasnet_5_large.cell_3(x_cell_1, x_cell_2)       # (1, 1080, 80, 60)
        # x_cell_4 = pnasnet_5_large.cell_4(x_cell_2, x_cell_3)       # (1, 2160, 40, 30)
        # x_cell_5 = pnasnet_5_large.cell_5(x_cell_3, x_cell_4)       # (1, 2160, 40, 30)
        # x_cell_6 = pnasnet_5_large.cell_6(x_cell_4, x_cell_5)       # (1, 2160, 40, 30)
        # x_cell_7 = pnasnet_5_large.cell_7(x_cell_5, x_cell_6)       # (1, 2160, 40, 30)
        # x_cell_8 = pnasnet_5_large.cell_8(x_cell_6, x_cell_7)       # (1, 4320, 20, 15)
        # x_cell_9 = pnasnet_5_large.cell_9(x_cell_7, x_cell_8)       # (1, 4320, 20, 15)
        # x_cell_10 = pnasnet_5_large.cell_10(x_cell_8, x_cell_9)     # (1, 4320, 20, 15)
        # x_cell_11 = pnasnet_5_large.cell_11(x_cell_9, x_cell_10)    # (1, 4320, 20, 15)

        conv1 = self.Conv1(pnasnet_5_large)
        conv2 = self.Conv2(pnasnet_5_large)
        conv3 = self.Conv3(pnasnet_5_large, conv1)
        conv4 = self.Conv4(pnasnet_5_large, conv3)
        conv5 = self.Conv5(pnasnet_5_large)

        num_conv1_out = 96
        num_conv2_out = 168
        num_conv3_out = 1008
        num_conv4_out = 2160
        num_conv5_out = 4320

        return Backbone.Component(
            conv1, conv2, conv3, conv4, conv5,
            num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out, num_conv5_out
        )

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.5, 0.5, 0.5

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.5, 0.5, 0.5