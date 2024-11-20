from typing import Tuple

import pretrainedmodels
from torch import nn, Tensor

from . import Backbone


class NASNet_A_Large(Backbone):

    class Conv1(nn.Module):

        def __init__(self, nasnet_a_large: nn.Module):
            super().__init__()
            self.conv0 = nasnet_a_large.conv0
            self.x_conv0 = None

        def forward(self, x: Tensor) -> Tensor:
            self.x_conv0 = self.conv0(x)
            return self.x_conv0

    class Conv2(nn.Module):

        def __init__(self, nasnet_a_large: nn.Module):
            super().__init__()
            self.cell_stem_0 = nasnet_a_large.cell_stem_0

        def forward(self, x_conv0: Tensor) -> Tensor:
            x_stem_0 = self.cell_stem_0(x_conv0)
            return x_stem_0

    class Conv3(nn.Module):

        def __init__(self, nasnet_a_large: nn.Module, conv1: 'NASNet_A_Large.Conv1'):
            super().__init__()
            self.conv1 = conv1
            self.cell_stem_1 = nasnet_a_large.cell_stem_1
            self.cell_0 = nasnet_a_large.cell_0
            self.cell_1 = nasnet_a_large.cell_1
            self.cell_2 = nasnet_a_large.cell_2
            self.cell_3 = nasnet_a_large.cell_3
            self.cell_4 = nasnet_a_large.cell_4
            self.cell_5 = nasnet_a_large.cell_5
            self.x_cell_4 = None

        def forward(self, x_stem_0: Tensor) -> Tensor:
            x_stem_1 = self.cell_stem_1(self.conv1.x_conv0, x_stem_0)
            x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
            x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
            x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
            x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
            self.x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
            x_cell_5 = self.cell_5(self.x_cell_4, x_cell_3)
            return x_cell_5

    class Conv4(nn.Module):

        def __init__(self, nasnet_a_large: nn.Module, conv3: 'NASNet_A_Large.Conv3'):
            super().__init__()
            self.conv3 = conv3
            self.reduction_cell_0 = nasnet_a_large.reduction_cell_0
            self.cell_6 = nasnet_a_large.cell_6
            self.cell_7 = nasnet_a_large.cell_7
            self.cell_8 = nasnet_a_large.cell_8
            self.cell_9 = nasnet_a_large.cell_9
            self.cell_10 = nasnet_a_large.cell_10
            self.cell_11 = nasnet_a_large.cell_11

        def forward(self, x_cell_5: Tensor) -> Tensor:
            x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, self.conv3.x_cell_4)
            x_cell_6 = self.cell_6(x_reduction_cell_0, self.conv3.x_cell_4)
            x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
            x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
            x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
            x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
            x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
            return x_cell_11

    class Conv5(nn.Module):

        def __init__(self, nasnet_a_large: nn.Module):
            super().__init__()
            self.reduction_cell_1 = nasnet_a_large.reduction_cell_1
            self.cell_12 = nasnet_a_large.cell_12
            self.cell_13 = nasnet_a_large.cell_13
            self.cell_14 = nasnet_a_large.cell_14
            self.cell_15 = nasnet_a_large.cell_15
            self.cell_16 = nasnet_a_large.cell_16
            self.cell_17 = nasnet_a_large.cell_17

        def forward(self, x_cell_11: Tensor) -> Tensor:
            x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_11)  # modify from `reduction_cell_1(x_cell_11, x_cell_10)` for compatibility
            x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_11)  # modify from `cell_12(x_reduction_cell_1, x_cell_10)` for compatibility
            x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
            x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
            x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
            x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
            x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
            return x_cell_17



    @staticmethod

    @staticmethod


    class Conv2(nn.Module):



    class Conv3(nn.Module):



    class Conv4(nn.Module):



    class Conv5(nn.Module):



    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        nasnet_a_large = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet' if self.pretrained else None)

        # x = torch.randn(1, 3, 640, 480)
        # x_conv0 = nasnet_a_large.conv0(x)                                           # (1, 96, 319, 239)
        # x_stem_0 = nasnet_a_large.cell_stem_0(x_conv0)                              # (1, 168, 160, 120)
        # x_stem_1 = nasnet_a_large.cell_stem_1(x_conv0, x_stem_0)                    # (1, 336, 80, 60)
        # x_cell_0 = nasnet_a_large.cell_0(x_stem_1, x_stem_0)                        # (1, 1008, 80, 60)
        # x_cell_1 = nasnet_a_large.cell_1(x_cell_0, x_stem_1)                        # (1, 1008, 80, 60)
        # x_cell_2 = nasnet_a_large.cell_2(x_cell_1, x_cell_0)                        # (1, 1008, 80, 60)
        # x_cell_3 = nasnet_a_large.cell_3(x_cell_2, x_cell_1)                        # (1, 1008, 80, 60)
        # x_cell_4 = nasnet_a_large.cell_4(x_cell_3, x_cell_2)                        # (1, 1008, 80, 60)
        # x_cell_5 = nasnet_a_large.cell_5(x_cell_4, x_cell_3)                        # (1, 1008, 80, 60)
        # x_reduction_cell_0 = nasnet_a_large.reduction_cell_0(x_cell_5, x_cell_4)    # (1, 1344, 40, 30)
        # x_cell_6 = nasnet_a_large.cell_6(x_reduction_cell_0, x_cell_4)              # (1, 2016, 40, 30)
        # x_cell_7 = nasnet_a_large.cell_7(x_cell_6, x_reduction_cell_0)              # (1, 2016, 40, 30)
        # x_cell_8 = nasnet_a_large.cell_8(x_cell_7, x_cell_6)                        # (1, 2016, 40, 30)
        # x_cell_9 = nasnet_a_large.cell_9(x_cell_8, x_cell_7)                        # (1, 2016, 40, 30)
        # x_cell_10 = nasnet_a_large.cell_10(x_cell_9, x_cell_8)                      # (1, 2016, 40, 30)
        # x_cell_11 = nasnet_a_large.cell_11(x_cell_10, x_cell_9)                     # (1, 2016, 40, 30)
        # x_reduction_cell_1 = nasnet_a_large.reduction_cell_1(x_cell_11, x_cell_10)  # (1, 2688, 20, 15)
        # x_cell_12 = nasnet_a_large.cell_12(x_reduction_cell_1, x_cell_10)           # (1, 4032, 20, 15)
        # x_cell_13 = nasnet_a_large.cell_13(x_cell_12, x_reduction_cell_1)           # (1, 4032, 20, 15)
        # x_cell_14 = nasnet_a_large.cell_14(x_cell_13, x_cell_12)                    # (1, 4032, 20, 15)
        # x_cell_15 = nasnet_a_large.cell_15(x_cell_14, x_cell_13)                    # (1, 4032, 20, 15)
        # x_cell_16 = nasnet_a_large.cell_16(x_cell_15, x_cell_14)                    # (1, 4032, 20, 15)
        # x_cell_17 = nasnet_a_large.cell_17(x_cell_16, x_cell_15)                    # (1, 4032, 20, 15)

        conv1 = self.Conv1(nasnet_a_large)
        conv2 = self.Conv2(nasnet_a_large)
        conv3 = self.Conv3(nasnet_a_large, conv1)
        conv4 = self.Conv4(nasnet_a_large, conv3)
        conv5 = self.Conv5(nasnet_a_large)

        num_conv1_out = 96
        num_conv2_out = 168
        num_conv3_out = 1008
        num_conv4_out = 2016
        num_conv5_out = 4032

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