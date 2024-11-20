"""
This script is used for evaluating the performance of YOLOv6 TensorRT models.
"""
import os
import sys
import json
import argparse
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tensorrt_processor import Processor

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])











if __name__ == '__main__':
    main()