import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.common import RealVGGBlock, LinearAddBlock
from torch.optim.sgd import SGD
from yolov6.utils.events import LOGGER












class RepVGGOptimizer(SGD):
    '''scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)'''



