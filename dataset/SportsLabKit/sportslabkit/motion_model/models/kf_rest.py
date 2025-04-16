from typing import Any

import numpy as np
import torch
from filterpy.kalman import predict, update
from numpy import ndarray

from sportslabkit.motion_model.base import BaseMotionModel


class KalmanFilter(BaseMotionModel):
    hparam_search_space: dict[str, dict[str, Any]] = {
        "dt": {"type": "categorical", "values": [10, 2, 1, 1 / 30, 1 / 60, 1 / 120]},
        "process_noise": {"type": "logfloat", "low": 1e-6, "high": 1e2},
        "measurement_noise": {"type": "logfloat", "low": 1e-3, "high": 1e2},
        "confidence_scaler": {"type": "logfloat", "low": 1e-3, "high": 100},
    }
    required_observation_types: list[str] = ["box", "score"]
    required_state_types: list[str] = ["x", "P", "F", "H", "R", "Q"]






