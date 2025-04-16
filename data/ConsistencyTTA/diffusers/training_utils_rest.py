import copy
import os
import random
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .utils import deprecate




    # ^^ safe to call this function even if cuda is not available


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """


    @classmethod



    @torch.no_grad()





