# Ultralytics YOLO 🚀, AGPL-3.0 license

import copy

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """
    Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of frames for computational efficiency.

    Attributes:
        method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.ndarray): Stores the previous frame for tracking.
        prevKeyPoints (list): Stores the keypoints from the previous frame.
        prevDescriptors (np.ndarray): Stores the descriptors from the previous frame.
        initializedFirstFrame (bool): Flag to indicate if the first frame has been processed.

    Methods:
        __init__(self, method='sparseOptFlow', downscale=2): Initializes a GMC object with the specified method
                                                              and downscale factor.
        apply(self, raw_frame, detections=None): Applies the chosen method to a raw frame and optionally uses
                                                 provided detections.
        applyEcc(self, raw_frame, detections=None): Applies the ECC algorithm to a raw frame.
        applyFeatures(self, raw_frame, detections=None): Applies feature-based methods like ORB or SIFT to a raw frame.
        applySparseOptFlow(self, raw_frame, detections=None): Applies the Sparse Optical Flow method to a raw frame.
    """





