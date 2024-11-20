from enum import Enum
from typing import Tuple, List, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor


class Algorithm(nn.Module):

    class Name(Enum):
        MASK_RCNN = 'mask_rcnn'

    OPTIONS = [it.value for it in Name]

    @staticmethod



