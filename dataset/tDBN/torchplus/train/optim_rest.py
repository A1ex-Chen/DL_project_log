from collections import defaultdict, Iterable

import torch
from copy import deepcopy
from itertools import chain
from torch.autograd import Variable

required = object()



class MixedPrecisionWrapper(object):
    """mixed precision optimizer wrapper.
    Arguments:
        optimizer (torch.optim.Optimizer): an instance of 
            :class:`torch.optim.Optimizer`
        scale: (float): a scalar for grad scale.
        auto_scale: (bool): whether enable auto scale.
            The algorihm of auto scale is discribled in 
            http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
    """








