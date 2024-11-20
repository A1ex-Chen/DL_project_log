from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from sportslabkit.checks import _check_cost_matrix, _check_detections, _check_trackers
from sportslabkit.metrics.object_detection import iou_score
from sportslabkit.types.detection import Detection
from sportslabkit.types.tracklet import Tracklet


class BaseCostMatrixMetric(ABC):
    """A base class for computing the cost matrix between trackers and
    detections."""


    @abstractmethod


class IoUCMM(BaseCostMatrixMetric):
    """Compute the IoU Cost Matrix Metric between trackers and detections."""




class EuclideanCMM(BaseCostMatrixMetric):
    """Compute the Euclidean Cost Matrix Metric between trackers and
    detections."""




# FIXME: 技術負債を返済しましょう
class EuclideanCMM2D(BaseCostMatrixMetric):



class CosineCMM(BaseCostMatrixMetric):
    """Compute the Cosine Cost Matrix Metric between trackers and
    detections."""
