import torch
import torch.nn as nn
from darts.api import Model
from darts.modules.conv.mixed_layer import MixedLayer
from darts.modules.operations.conv import ConvBlock, FactorizedReduce


class Cell(Model):
