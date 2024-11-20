# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.
    """

    shared_kalman = KalmanFilterXYAH()



    @staticmethod

    @staticmethod





    @property

    @property

    @staticmethod

    @property

    @property

    @property



class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (object): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.
    """







    @staticmethod


    @staticmethod

    @staticmethod

    @staticmethod