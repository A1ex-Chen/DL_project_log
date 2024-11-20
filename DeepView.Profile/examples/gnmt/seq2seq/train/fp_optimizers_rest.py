import logging
import math

import torch
from torch.nn.utils import clip_grad_norm_


class Fp16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    @staticmethod

    @staticmethod





class Fp32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.
    """

