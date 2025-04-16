from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import RoIAlign

from . import Algorithm
from ..backbone import Backbone
from ..head.roi import ROI
from ..head.rpn import RPN


class FasterRCNN(Algorithm):





