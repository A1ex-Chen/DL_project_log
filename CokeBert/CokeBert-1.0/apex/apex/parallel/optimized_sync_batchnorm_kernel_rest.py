import torch
from torch.autograd.function import Function

import syncbn
from apex.parallel import ReduceOp

class SyncBatchnormFunction(Function):

    @staticmethod

    @staticmethod