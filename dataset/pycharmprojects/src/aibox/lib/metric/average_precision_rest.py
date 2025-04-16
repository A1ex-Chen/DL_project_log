import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class AveragePrecision:

    @dataclass
    class Result:
        ap: float
        inter_recall_array: np.ndarray
        inter_precision_array: np.ndarray
        recall_array: np.ndarray
        precision_array: np.ndarray
        accuracy_array: np.ndarray
        prob_array: np.ndarray

    @dataclass
    class PyCOCOToolsResult:
        mean_mean_ap: float
        mean_standard_ap: float
        mean_strict_ap: float



    @staticmethod


    @staticmethod

    @staticmethod