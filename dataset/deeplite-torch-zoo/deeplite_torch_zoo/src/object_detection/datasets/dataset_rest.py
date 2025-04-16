# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from deeplite_torch_zoo.utils import LOGGER, LOCAL_RANK, NUM_THREADS, is_dir_writeable, TQDM_BAR_FORMAT
from deeplite_torch_zoo.src.object_detection.datasets.base import BaseDataset
from deeplite_torch_zoo.src.object_detection.datasets.utils import get_hash, img2label_paths, verify_image_label
from deeplite_torch_zoo.src.object_detection.datasets.augment import Compose, Format, LetterBox, \
    v8_transforms as image_transforms
from deeplite_torch_zoo.src.object_detection.datasets.instance import Instances


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    cache_version = '1.0.2'  # dataset labels *.cache version, >= 1.0.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]




    # TODO: use hyp config to set all these augmentations



    @staticmethod