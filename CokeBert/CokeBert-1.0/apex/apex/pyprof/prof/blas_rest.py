from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase
import numpy as np

class Addmm(OperatorLayerBase):








class Bmm(OperatorLayerBase):








class Matmul(OperatorLayerBase):

	NON_GEMM = ["kernelPointwiseApply2", "reduce_1Block_kernel", "elementwise_kernel"]
	NON_TC = NON_GEMM + ["dot_kernel"]








class Mm(OperatorLayerBase):






