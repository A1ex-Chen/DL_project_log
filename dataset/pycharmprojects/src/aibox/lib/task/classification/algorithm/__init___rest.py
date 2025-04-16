from enum import Enum
from typing import Tuple, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor


class Algorithm(nn.Module):

    class Name(Enum):
        MOBILENET_V2 = 'mobilenet_v2'
        GOOGLENET = 'googlenet'
        INCEPTION_V3 = 'inception_v3'
        RESNET18 = 'resnet18'
        RESNET34 = 'resnet34'
        RESNET50 = 'resnet50'
        RESNET101 = 'resnet101'
        EFFICIENTNET_B0 = 'efficientnet_b0'
        EFFICIENTNET_B1 = 'efficientnet_b1'
        EFFICIENTNET_B2 = 'efficientnet_b2'
        EFFICIENTNET_B3 = 'efficientnet_b3'
        EFFICIENTNET_B4 = 'efficientnet_b4'
        EFFICIENTNET_B5 = 'efficientnet_b5'
        EFFICIENTNET_B6 = 'efficientnet_b6'
        EFFICIENTNET_B7 = 'efficientnet_b7'
        RESNEST50 = 'resnest50'
        RESNEST101 = 'resnest101'
        RESNEST200 = 'resnest200'
        RESNEST269 = 'resnest269'
        REGNET_Y_400MF = 'regnet_y_400mf'

    OPTIONS = [it.value for it in Name]

    @staticmethod







    @property

    @property

    @staticmethod

    @staticmethod