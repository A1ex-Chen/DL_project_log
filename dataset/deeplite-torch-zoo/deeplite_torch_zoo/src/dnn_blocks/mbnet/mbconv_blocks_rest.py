import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import (
    ConvBnAct,
    DWConv,
    get_activation,
    round_channels,
)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import SELayer


class MBConv(nn.Module):
    # Taken from: https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
