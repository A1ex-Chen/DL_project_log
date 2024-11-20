# Copyright (c) Facebook, Inc. and its affiliates.
import io
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import CfgNode, instantiate
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances, ROIMasks
from detectron2.utils.file_io import PathManager


"""
Internal utilities for tests. Don't use except for writing tests.
"""











