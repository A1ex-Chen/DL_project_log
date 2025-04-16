"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple

import networkx as nx
import numpy as np

from sportslabkit import Tracklet
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection


EPS = 1e-7
# Define the named tuple outside of the function.
Node = namedtuple("Node", ["frame", "detection", "is_dummy"])


class BaseBatchMatchingFunction:
    """A base class for batch matching functions.

    A batch matching function takes a list of trackers and a list of detections
    and returns a list of matches.
    """


    @abstractmethod

    @abstractmethod