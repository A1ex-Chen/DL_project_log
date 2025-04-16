import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer


class FusedMBConv(nn.Module):
    # Taken from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
