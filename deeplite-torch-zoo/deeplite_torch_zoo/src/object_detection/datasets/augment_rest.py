# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy

import cv2
import torch
import numpy as np

from deeplite_torch_zoo.utils import LOGGER, check_version, colorstr
from deeplite_torch_zoo.src.object_detection.datasets.utils import polygons2masks, polygons2masks_overlap, \
    bbox_ioa, segment2box
from deeplite_torch_zoo.src.object_detection.datasets.instance import Instances




class RandomFlip:




class RandomHSV:




class Albumentations:
    """YOLOv8 Albumentations class (optional, only used if package is installed)"""




class RandomPerspective:









class CopyPaste:




class BaseMixTransform:
    """This implementation is from mmyolo."""






class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """






    @staticmethod



class MixUp(BaseMixTransform):





class Compose:







class Format:






class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""


