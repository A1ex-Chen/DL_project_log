# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# Based on ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.timm import DropPath


class LargeKernelReparam(nn.Module):



class MLP(nn.Module):



class RepLKBlock(nn.Module):
