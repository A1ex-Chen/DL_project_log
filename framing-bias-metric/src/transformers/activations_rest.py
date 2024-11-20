import math

import torch
import torch.nn.functional as F
from packaging import version

from .utils import logging


logger = logging.get_logger(__name__)






if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = F.gelu






if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = F.silu






ACT2FN = {
    "relu": F.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}

