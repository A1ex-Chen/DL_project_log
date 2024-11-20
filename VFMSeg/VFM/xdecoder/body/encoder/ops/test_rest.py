# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H*W).item() for H, W in shapes])


torch.manual_seed(3)


@torch.no_grad()


@torch.no_grad()




if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()

    for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
        check_gradient_numerical(channels, True, True, True)


