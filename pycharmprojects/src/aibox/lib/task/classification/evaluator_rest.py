from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Dataset, ConcatDataset
from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch


class Evaluator:

    @dataclass
    class Prediction:
        sorted_all_image_ids: List[str]
        sorted_all_pred_classes: Tensor
        sorted_all_pred_probs: Tensor
        sorted_all_gt_classes: Tensor

    @dataclass
    class Evaluation:

        @dataclass
        class MetricResult:
            mean_value: float
            class_to_value_dict: Dict[int, float]

        accuracy: float
        avg_recall: float
        avg_precision: float
        avg_f1_score: float
        confusion_matrix: np.ndarray
        class_to_fpr_array_dict: Dict[int, np.ndarray]
        class_to_tpr_array_dict: Dict[int, np.ndarray]
        class_to_thresh_array_dict: Dict[int, np.ndarray]
        metric_auc: MetricResult
        metric_sensitivity: MetricResult
        metric_specificity: MetricResult


    @torch.no_grad()

