# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from PIL import Image
from typing import Union, Optional




class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """






class SemSegEvaluatorParts(SemSegEvaluator):

        #print(self.n_merged_cls,"FFF")
    

