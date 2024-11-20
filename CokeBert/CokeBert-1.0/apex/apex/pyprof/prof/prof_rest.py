#!/usr/bin/env python3

"""
This script reads the output (Python dictionary) created by parse.py.
For every kernel (line) in the input it determines
	module / class name e.g. torch.nn.functional
	operator name e.g. linear
	kernel parameters e.g. GEMM M, N, K, datatype
	bytes
	flops
	tensor core usage
	direction (fprop, bprop)
	and other things. Please see the tool usage.
"""

from .usage import parseArgs
from .output import Output
from .utility import Utility
from .pointwise import Pointwise
from .convert import Convert
from .blas import *
from .embedding import Embedding
from .reduction import *
from .dropout import Dropout
from .softmax import *
#from pooling import * # work in progress
from .linear import Linear
from .optim import Adam
from .misc import *
from .conv import Conv
from .activation import Activation
from .index_slice_join_mutate import Cat, Reshape, MaskedScatter, Gather, Nonzero, IndexSelect, MaskedSelect
from .recurrentCell import RNNCell
from .normalization import BatchNorm
from .randomSample import RandPerm
from .loss import MSELoss
from .data import Data

	#print("Error: seqId {} not found.".format(seq), file=sys.stderr)
	#assert False



kernels = []
if __name__ == '__main__':
	main()