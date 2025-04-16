"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from sportslabkit import Tracklet
from sportslabkit.checks import (
    _check_cost_matrix,
    _check_detections,
    _check_matches,
    _check_trackers,
)
from sportslabkit.types.detection import Detection


EPS = 1e-7




class BaseMatchingFunction(ABC):
    """A base class for matching functions.

    A matching function takes a list of trackers and a list of
    detections and returns a list of matches. Subclasses should
    implement the :meth:`compute_cost_matrix` method.
    """


    @abstractmethod
