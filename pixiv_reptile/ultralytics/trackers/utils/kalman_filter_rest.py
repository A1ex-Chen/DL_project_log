# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    For bytetrack. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect
    ratio a, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct
    observation of the state space (linear observation model).
    """









class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    For BoT-SORT. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains the bounding box center position (x, y), width
    w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).
    """




