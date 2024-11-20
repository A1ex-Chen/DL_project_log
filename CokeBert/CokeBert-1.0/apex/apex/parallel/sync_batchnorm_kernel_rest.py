import torch
from torch.autograd.function import Function

from apex.parallel import ReduceOp


class SyncBatchnormFunction(Function):

    @staticmethod

    @staticmethod