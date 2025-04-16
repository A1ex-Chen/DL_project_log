from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sportslabkit import Tracklet


class BaseMotionModel(ABC):
    """Abstract base class for motion models.

    This class defines a common interface for all motion models.
    Derived classes should implement the update, and predict methods. MotionModels are procedural and stateless. The state of tracklet is managed by the Tracklet class. The tracklet must have the required observations and states for the motion model to work. If the tracklet doesn't have the required observations or states, the motion model will raise an error and tell the user which observations or states are missing.
    """

    hparam_search_space: dict[str, type] = {}
    required_observation_types: list[str] = NotImplemented
    required_state_types: list[str] = NotImplemented




    @abstractmethod

    @classmethod

