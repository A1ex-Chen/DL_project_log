import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import Backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage

from ..postprocessing import detector_postprocess




class DenseDetector(nn.Module):
    """
    Base class for dense detector. We define a dense detector as a fully-convolutional model that
    makes per-pixel (i.e. dense) predictions.
    """


    @property







