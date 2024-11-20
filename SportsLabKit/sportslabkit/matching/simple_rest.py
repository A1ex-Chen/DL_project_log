from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import DefaultDict

import networkx as nx
import numpy as np

from sportslabkit import Tracklet
from sportslabkit.matching.base import BaseMatchingFunction
from sportslabkit.matching.base_batch import BaseBatchMatchingFunction, Node
from sportslabkit.metrics import BaseCostMatrixMetric, IoUCMM
from sportslabkit.types.detection import Detection


class SimpleMatchingFunction(BaseMatchingFunction):
    """A matching function that uses a single metric.

    Args:
        metric: A metric. Defaults to `IoUCMM`.
        gate: The gate of the metric, i.e. if the metric is larger than
            this value, the cost will be set to infinity. Defaults to `np.inf`.

    Note:
        To implement your own matching function, you can inherit from `BaseMatchingFunction`
        and override the :meth:`compute_cost_matrix` method.
    """




class SimpleBatchMatchingFunction(BaseBatchMatchingFunction):
    """A batch matching function that uses a simple distance metric.

    This class is a simple implementation of batch matching function where the cost is based on the Euclidean distance between the trackers and detections.
    """

