import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Optional, Tuple, Union
import torch
from torch import nn
from .norm import NORM_CLASS_REGISTRY












MODEL_INIT_REGISTRY = {'default_': torch_default_param_init_fn_, 'baseline_': baseline_param_init_fn_, 'kaiming_uniform_': kaiming_uniform_param_init_fn_, 'kaiming_normal_': kaiming_normal_param_init_fn_, 'neox_init_': neox_param_init_fn_, 'small_init_': small_param_init_fn_, 'xavier_uniform_': xavier_uniform_param_init_fn_, 'xavier_normal_': xavier_normal_param_init_fn_}