import torch
import torch.nn as nn
import copy

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from tools.partial_quantization.utils import set_module, module_quant_disable












