import math
from functools import reduce

import numpy as np
import torch
from torch import FloatTensor as FTensor
from torch import stack as tstack

import torchplus
from torchplus.tools import torch_to_np_dtype
from tDBN.core.box_np_ops import iou_jit
from tDBN.core.non_max_suppression.nms_gpu import (nms_gpu, rotate_iou_gpu,
                                                       rotate_nms_gpu)
from tDBN.core.non_max_suppression.nms_cpu import rotate_nms_cc



    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)




    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)
































