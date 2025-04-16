from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Type

from torch import nn


class Backbone:

    class Name(Enum):
        MOBILENET_V3_SMALL = 'mobilenet_v3_small'
        MOBILENET_V3_LARGE = 'mobilenet_v3_large'
        RESNET18 = 'resnet18'
        RESNET34 = 'resnet34'
        RESNET50 = 'resnet50'
        RESNET101 = 'resnet101'
        RESNET152 = 'resnet152'
        RESNEXT50_32X4D = 'resnext50_32x4d'
        RESNEXT101_32X8D = 'resnext101_32x8d'
        WIDE_RESNET50_2 = 'wide_resnet50_2'
        WIDE_RESNET101_2 = 'wide_resnet101_2'
        SENET154 = 'senet154'
        SE_RESNEXT50_32X4D = 'se_resnext50_32x4d'
        SE_RESNEXT101_32X4D = 'se_resnext101_32x4d'
        NASNET_A_LARGE = 'nasnet_a_large'
        PNASNET_5_LARGE = 'pnasnet_5_large'
        RESNEST50 = 'resnest50'
        RESNEST101 = 'resnest101'
        RESNEST200 = 'resnest200'
        RESNEST269 = 'resnest269'
        RegNet_Y_8GF = 'regnet_y_8gf'
        EFFICIENTNET_B7 = 'efficientnet_b7'
        ConvNeXt_Base = 'convnext_base'
        EfficientNet_V2 = 'efficientnet_v2'

    OPTIONS = [it.value for it in Name]

    @staticmethod


    @dataclass
    class Component:
        conv1: nn.Module
        conv2: nn.Module
        conv3: nn.Module
        conv4: nn.Module
        conv5: nn.Module
        num_conv1_out: int
        num_conv2_out: int
        num_conv3_out: int
        num_conv4_out: int
        num_conv5_out: int



    @staticmethod

    @staticmethod