import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT

try:
  import amp_C
  from amp_C import multi_tensor_axpby
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorAxpby(unittest.TestCase):



    # The tensor creation here is written for convenience, not speed.

    # def find_inf(self, sizea, sizeb, applier, repeat_tensors, in_type, out_type, t, ind, val, inplace=False):
    #     self.overflow_buf.zero_()
    #     a = torch.cuda.FloatTensor(sizea).fill_(self.scale)
    #     b = torch.cuda.FloatTensor(sizeb).fill_(self.scale)

    #     out_list = []
    #     for i in range(repeat_tensors):
    #         out_list += [a.clone().to(out_type), b.clone().to(out_type)]

    #     if inplace:
    #         in_list = out_list
    #     else:
    #         in_list = [out.clone().to(in_type) for out in out_list]

    #     applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)

    #     self.overflow_buf.zero_()
    #     in_list[t][ind] = val
    #     applier(multi_tensor_scale, self.overflow_buf, [in_list, out_list], 1./self.scale)
    #     self.assertTrue(self.overflow_buf.item())

    @unittest.skipIf(disabled, "amp_C is unavailable")
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #               0, 0, float('nan'), inplace=inplace)
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #               2*repeat-1, sizeb-1, float('inf'), inplace=inplace)
                      # self.find_inf(sizea, sizeb, applier, repeat, in_type, out_type,
                      #              2*(repeat//2), sizea//2, float('inf'), inplace=inplace)



if __name__ == '__main__':
    unittest.main()