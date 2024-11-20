from enum import Enum
from typing import Tuple, Optional, List

import albumentations as A
import numpy as np
import torch
from imgaug import BoundingBox, BoundingBoxesOnImage, SegmentationMapsOnImage
from imgaug.augmenters.arithmetic import Add, SaltAndPepper
from imgaug.augmenters.blur import GaussianBlur
from imgaug.augmenters.color import AddToHueAndSaturation, Grayscale
from imgaug.augmenters.contrast import LogContrast
from imgaug.augmenters.convolutional import Sharpen
from imgaug.augmenters.flip import Fliplr, Flipud
from imgaug.augmenters.geometric import Affine, Rot90
from imgaug.augmenters.meta import Sometimes, Sequential, OneOf, SomeOf
from imgaug.augmenters.size import Crop
from torch import Tensor


class Augmenter:

    class Strategy(Enum):
        ALL = 'all'
        ONE = 'one'
        SOME = 'some'

    OPTIONS = [it.value for it in Strategy]





    @staticmethod