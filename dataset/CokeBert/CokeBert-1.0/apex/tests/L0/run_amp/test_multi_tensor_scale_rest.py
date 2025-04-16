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
  from amp_C import multi_tensor_scale 
  from apex.multi_tensor_apply import MultiTensorApply
  disabled = False
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


class TestMultiTensorScale(unittest.TestCase):



    # The tensor creation here is written for convenience, not speed.
 

    # Currently, the fused kernel gives a hard error if you attempt to downscale
    # into fp16 output, which imo is the desired behavior.  Maybe someday we
    # will learn otherwise.
    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp16_to_fp16(self):
    #     self.downscale(self.fp16, self.fp16, self.fp16_ref)
    # 
    # @unittest.skipIf(disabled, "amp_C is unavailable")
    # def test_fp32_to_fp16(self):
    #     self.downscale(self.fp32, self.fp16, self.fp16_ref)

    @unittest.skipIf(disabled, "amp_C is unavailable")



if __name__ == '__main__':
    unittest.main()