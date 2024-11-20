# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import lru_cache
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

from detectron2.utils.develop import create_dummy_class, create_dummy_func

from .wrappers import _NewEmptyTensorOp


class _DeformConv(Function):
    @staticmethod

    @staticmethod
    @once_differentiable

    @staticmethod

    @staticmethod
    @lru_cache(maxsize=128)


class _ModulatedDeformConv(Function):
    @staticmethod

    @staticmethod
    @once_differentiable

    @staticmethod


deform_conv = _DeformConv.apply
modulated_deform_conv = _ModulatedDeformConv.apply


class DeformConv(nn.Module):




class ModulatedDeformConv(nn.Module):




try:
    from detectron2 import _C
except ImportError:
    # TODO: register ops natively so there is no need to import _C.
    _msg = "detectron2 is not compiled successfully, please build following the instructions!"
    _args = ("detectron2._C", _msg)
    DeformConv = create_dummy_class("DeformConv", *_args)
    ModulatedDeformConv = create_dummy_class("ModulatedDeformConv", *_args)
    deform_conv = create_dummy_func("deform_conv", *_args)
    modulated_deform_conv = create_dummy_func("modulated_deform_conv", *_args)