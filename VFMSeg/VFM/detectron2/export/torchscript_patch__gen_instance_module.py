def _gen_instance_module(fields):
    s = """
from copy import deepcopy
import torch
from torch import Tensor
import typing
from typing import *

import detectron2
from detectron2.structures import Boxes, Instances

"""
    cls_name, cls_def = _gen_instance_class(fields)
    s += cls_def
    return cls_name, s
