import contextlib
import copy
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .models import UNet2DConditionModel
from .schedulers import SchedulerMixin
from .utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    deprecate,
    is_peft_available,
    is_torch_npu_available,
    is_torchvision_available,
    is_transformers_available,
)


if is_transformers_available():
    import transformers

if is_peft_available():
    from peft import set_peft_model_state_dict

if is_torchvision_available():
    from torchvision import transforms

if is_torch_npu_available():
    import torch_npu  # noqa: F401


        # ^^ safe to call this function even if cuda is not available














# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """


    @classmethod



    @torch.no_grad()





