from typing import Any

import numpy as np

from sportslabkit.motion_model.base import BaseMotionModel


class ExponentialMovingAverage(BaseMotionModel):
    """
    Exponential Moving Average (EMA) motion model for object tracking.

    This class implements an EMA-based motion model for object tracking.
    It can be used both in a stateful and a procedural manner.

    Attributes:
        gamma (float): The weight for the exponential moving average calculation.
        _value (Union[float, np.ndarray, None]): The internal state of the motion model.
    """

    hparam_search_space: dict[str, dict[str, object]] = {"gamma": {"type": "float", "low": 0.0, "high": 1.0}}
    required_observation_types = ["box"]
    required_state_types = ["EMA_t"]

