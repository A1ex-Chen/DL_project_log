import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from tDBN.core import box_torch_ops
from tDBN.core.losses import (WeightedSigmoidClassificationLoss,
                              WeightedSmoothL1LocalizationLoss,
                              WeightedSoftmaxClassificationLoss)
import operator
import torch
import warnings
from torch.nn.parallel.data_parallel import *

class tDBN_1(nn.Module):




class tDBN_2(nn.Module):






class tDBN_bv_1(nn.Module):





class tDBN_bv_2(nn.Module):





