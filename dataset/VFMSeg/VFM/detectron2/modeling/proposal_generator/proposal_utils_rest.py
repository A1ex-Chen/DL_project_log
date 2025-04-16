# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple, Union
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)







