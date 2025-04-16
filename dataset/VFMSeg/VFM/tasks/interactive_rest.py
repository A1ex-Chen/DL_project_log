# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from xdecoder.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

