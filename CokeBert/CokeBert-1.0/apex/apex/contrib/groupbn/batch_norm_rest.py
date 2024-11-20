import torch
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

import bnp

class bn_NHWC_impl(torch.autograd.Function):
    @staticmethod

    @staticmethod


class bn_addrelu_NHWC_impl(torch.autograd.Function):
    @staticmethod

    @staticmethod





class BatchNorm2d_NHWC(_BatchNorm):
    # if using BatchNorm2d_NHWC simultaneously with multiple streams set multi_stream to True


