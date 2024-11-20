import torch

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, _create_grid_offsets
from detectron2.modeling import ANCHOR_GENERATOR_REGISTRY
from detectron2.structures import Boxes
import math 
import detectron2.utils.comm as comm


@ANCHOR_GENERATOR_REGISTRY.register()
class AnchorGeneratorWithCenter(DefaultAnchorGenerator):

