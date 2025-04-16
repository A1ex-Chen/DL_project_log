import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch
import warnings
from torch import nn

try:
    from pytorch_quantization import nn as quant_nn
except ImportError as e:
    warnings.warn(
        "pytorch_quantization module not found, quantization will not be available"
    )
    quant_nn = None


# LayerBuilder {{{
class LayerBuilder(object):
    @dataclass
    class Config:
        activation: str = "relu"
        conv_init: str = "fan_in"
        bn_momentum: Optional[float] = None
        bn_epsilon: Optional[float] = None











# LayerBuilder }}}

# LambdaLayer {{{
class LambdaLayer(nn.Module):



# }}}

# SqueezeAndExcitation {{{
class SqueezeAndExcitation(nn.Module):



# }}}

# EMA {{{
class EMA:






# }}}

# ONNXSiLU {{{
# Since torch.nn.SiLU is not supported in ONNX,
# it is required to use this implementation in exported model (15-20% more GPU memory is needed)
class ONNXSiLU(nn.Module):



# }}}


class SequentialSqueezeAndExcitation(SqueezeAndExcitation):
