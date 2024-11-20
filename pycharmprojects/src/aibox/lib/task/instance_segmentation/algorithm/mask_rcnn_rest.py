from typing import List, Union, Tuple

import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator as AnchorGenerator_

from . import Algorithm


class AnchorGenerator(AnchorGenerator_):



class MaskRCNN(Algorithm):


