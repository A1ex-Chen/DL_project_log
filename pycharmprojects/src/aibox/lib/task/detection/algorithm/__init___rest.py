from enum import Enum
from typing import Tuple, List, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor

from ..backbone import Backbone


class Algorithm(nn.Module):

    class Name(Enum):
        FASTER_RCNN = 'faster_rcnn'
        FPN = 'fpn'
        TORCH_FPN = 'torch_fpn'

    OPTIONS = [it.value for it in Name]

    @staticmethod




