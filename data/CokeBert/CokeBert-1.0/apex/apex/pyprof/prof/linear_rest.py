from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Linear(OperatorLayerBase):

	'''
	Notes:
	If the bias occurs before the GEMM, then its 1 write (bias expansion).
	If the bias occurs after, then its 1 read and 1 write.
	bias in bprop is a reduction and hence is 1 read.
	'''

	gemmKernels = ["gemm", "gemv", "dot_kernel", "splitKreduce_kernel", "reduce_1Block_kernel"]
	biasKernels = ["kernelReduceContigDim", "kernelReduceNoncontigDim_shared", "elementwise_kernel", "reduce_kernel"]








