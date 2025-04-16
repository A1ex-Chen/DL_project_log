# Taken from https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py
# The file is modified by Deeplite Inc. from the original implementation on Dec 22, 2022
# Blocks implementation refactoring

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import autopad, get_activation
from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS
from deeplite_torch_zoo.utils import LOGGER


@VARIABLE_CHANNEL_BLOCKS.register()
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697







