from typing import Union, Tuple

import torchvision.models
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Backbone


class EfficientNet_V2(Backbone):



    @staticmethod

    @staticmethod