from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sportslabkit import Tracklet
from sportslabkit.matching.base import BaseMatchingFunction
from sportslabkit.metrics import BaseCostMatrixMetric, CosineCMM, IoUCMM
from sportslabkit.types.detection import Detection


class MotionVisualMatchingFunction(BaseMatchingFunction):
    """A matching function that uses a combination of motion and visual
    metrics.

    Args:
        motion_metric: A motion metric. Defaults to `IoUCMM`.
        beta: The weight of the motion metric. The weight of the visual metric is calculated as 1 - beta. Defaults to 0.5.
        motion_metric_gate: The gate of the motion metric, i.e. if the
            motion metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.
        visual_metric: A visual metric. Defaults to `CosineCMM`.
        visual_metric_gate: The gate of the visual metric, i.e. if the
            visual metric is larger than this value, the cost will be
            set to infinity. Defaults to `np.inf`.

    Note:
        To implement your own matching function, you can inherit from `BaseMatchingFunction`
        and override the :meth:`compute_cost_matrix` method.
    """

    hparam_search_space = {
        "beta": {"type": "float", "low": 0, "high": 1},
        "motion_metric_gate": {"type": "logfloat", "low": 1e-3, "high": 1e2},
        "visual_metric_gate": {"type": "logfloat", "low": 1e-3, "high": 1e2},
    }

