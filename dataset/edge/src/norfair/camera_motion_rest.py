"Camera motion stimation module."
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()


#
# Abstract interfaces
#
class CoordinatesTransformation(ABC):
    """
    Abstract class representing a coordinate transformation.

    Detections' and tracked objects' coordinates can be interpreted in 2 reference:

    - _Relative_: their position on the current frame, (0, 0) is top left
    - _Absolute_: their position on an fixed space, (0, 0)
        is the top left of the first frame of the video.

    Therefore, coordinate transformation in this context is a class that can transform
    coordinates in one reference to another.
    """

    @abstractmethod
    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        pass


class TransformationGetter(ABC):
    """
    Abstract class representing a method for finding CoordinatesTransformation between 2 sets of points
    """

    @abstractmethod
    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> Tuple[bool, CoordinatesTransformation]:
        pass


#
# Translation
#
class TranslationTransformation(CoordinatesTransformation):
    """
    Coordinate transformation between points using a simple translation

    Parameters
    ----------
    movement_vector : np.ndarray
        The vector representing the translation.
    """

    def __init__(self, movement_vector):
        self.movement_vector = movement_vector

    def abs_to_rel(self, points: np.ndarray):
        return points + self.movement_vector

    def rel_to_abs(self, points: np.ndarray):
        return points - self.movement_vector


class TranslationTransformationGetter(TransformationGetter):
    """
    Calculates TranslationTransformation between points.

    The camera movement is calculated as the mode of optical flow between the previous reference frame
    and the current.

    Comparing consecutive frames can make differences too small to correctly estimate the translation,
    for this reason the reference frame is kept fixed as we progress through the video.
    Eventually, if the transformation is no longer able to match enough points, the reference frame is updated.

    Parameters
    ----------
    bin_size : float
        Before calculatin the mode, optiocal flow is bucketized into bins of this size.
    proportion_points_used_threshold: float
        Proportion of points that must be matched, otherwise the reference frame must be updated.
    """

    def __init__(
        self, bin_size: float = 0.2, proportion_points_used_threshold: float = 0.9
    ) -> None:
        self.bin_size = bin_size
        self.proportion_points_used_threshold = proportion_points_used_threshold
        self.data = None

    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> Tuple[bool, TranslationTransformation]:
        # get flow
        flow = curr_pts - prev_pts

        # get mode
        flow = np.around(flow / self.bin_size) * self.bin_size
        unique_flows, counts = np.unique(flow, axis=0, return_counts=True)

        max_index = counts.argmax()

        proportion_points_used = counts[max_index] / len(prev_pts)
        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        flow_mode = unique_flows[max_index]

        try:
            flow_mode += self.data
        except TypeError:
            pass

        if update_prvs:
            self.data = flow_mode

        return update_prvs, TranslationTransformation(flow_mode)


#
# Homography
#
class HomographyTransformation(CoordinatesTransformation):
    """
    Coordinate transformation beweent points using an homography

    Parameters
    ----------
    homography_matrix : np.ndarray
        The matrix representing the homography
    """

    def __init__(self, homography_matrix: np.ndarray):
        self.homography_matrix = homography_matrix
        self.inverse_homography_matrix = np.linalg.inv(homography_matrix)

    def abs_to_rel(self, points: np.ndarray):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]

    def rel_to_abs(self, points: np.ndarray):
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.inverse_homography_matrix.T
        points_transformed = points_transformed / points_transformed[:, -1].reshape(
            -1, 1
        )
        return points_transformed[:, :2]


class HomographyTransformationGetter(TransformationGetter):
    """
    Calculates HomographyTransformation between points.

    The camera movement is represented as an homography that matches the optical flow between the previous reference frame
    and the current.

    Comparing consecutive frames can make differences too small to correctly estimate the homography, often resulting in the identity.
    For this reason the reference frame is kept fixed as we progress through the video.
    Eventually, if the transformation is no longer able to match enough points, the reference frame is updated.

    Parameters
    ----------
    method : Optional[int], optional
        One of openCV's method for finding homographies.
        Valid options are: `[0, cv.RANSAC, cv.LMEDS, cv.RHO]`, by default `cv.RANSAC`
    ransac_reproj_threshold : int, optional
        Maximum allowed reprojection error to treat a point pair as an inlier. More info in links below.
    max_iters : int, optional
        The maximum number of RANSAC iterations.  More info in links below.
    confidence : float, optional
        Confidence level, must be between 0 and 1. More info in links below.
    proportion_points_used_threshold : float, optional
        Proportion of points that must be matched, otherwise the reference frame must be updated.

    See Also
    --------
    [opencv.findHomography](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)
    """

    def __init__(
        self,
        method: Optional[int] = None,
        ransac_reproj_threshold: int = 3,
        max_iters: int = 2000,
        confidence: float = 0.995,
        proportion_points_used_threshold: float = 0.9,
    ) -> None:
        self.data = None
        if method is None:
            method = cv2.RANSAC
        self.method = method
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.proportion_points_used_threshold = proportion_points_used_threshold

    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> Tuple[bool, HomographyTransformation]:
        homography_matrix, points_used = cv2.findHomography(
            prev_pts,
            curr_pts,
            method=self.method,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        proportion_points_used = np.sum(points_used) / len(points_used)

        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        try:
            homography_matrix = homography_matrix @ self.data
        except (TypeError, ValueError):
            pass

        if update_prvs:
            self.data = homography_matrix

        return update_prvs, HomographyTransformation(homography_matrix)


#
# Motion estimation
#

    @abstractmethod


class TransformationGetter(ABC):
    """
    Abstract class representing a method for finding CoordinatesTransformation between 2 sets of points
    """

    @abstractmethod


#
# Translation
#
class TranslationTransformation(CoordinatesTransformation):
    """
    Coordinate transformation between points using a simple translation

    Parameters
    ----------
    movement_vector : np.ndarray
        The vector representing the translation.
    """





class TranslationTransformationGetter(TransformationGetter):
    """
    Calculates TranslationTransformation between points.

    The camera movement is calculated as the mode of optical flow between the previous reference frame
    and the current.

    Comparing consecutive frames can make differences too small to correctly estimate the translation,
    for this reason the reference frame is kept fixed as we progress through the video.
    Eventually, if the transformation is no longer able to match enough points, the reference frame is updated.

    Parameters
    ----------
    bin_size : float
        Before calculatin the mode, optiocal flow is bucketized into bins of this size.
    proportion_points_used_threshold: float
        Proportion of points that must be matched, otherwise the reference frame must be updated.
    """




#
# Homography
#
class HomographyTransformation(CoordinatesTransformation):
    """
    Coordinate transformation beweent points using an homography

    Parameters
    ----------
    homography_matrix : np.ndarray
        The matrix representing the homography
    """





class HomographyTransformationGetter(TransformationGetter):
    """
    Calculates HomographyTransformation between points.

    The camera movement is represented as an homography that matches the optical flow between the previous reference frame
    and the current.

    Comparing consecutive frames can make differences too small to correctly estimate the homography, often resulting in the identity.
    For this reason the reference frame is kept fixed as we progress through the video.
    Eventually, if the transformation is no longer able to match enough points, the reference frame is updated.

    Parameters
    ----------
    method : Optional[int], optional
        One of openCV's method for finding homographies.
        Valid options are: `[0, cv.RANSAC, cv.LMEDS, cv.RHO]`, by default `cv.RANSAC`
    ransac_reproj_threshold : int, optional
        Maximum allowed reprojection error to treat a point pair as an inlier. More info in links below.
    max_iters : int, optional
        The maximum number of RANSAC iterations.  More info in links below.
    confidence : float, optional
        Confidence level, must be between 0 and 1. More info in links below.
    proportion_points_used_threshold : float, optional
        Proportion of points that must be matched, otherwise the reference frame must be updated.

    See Also
    --------
    [opencv.findHomography](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)
    """




#
# Motion estimation
#
def _get_sparse_flow(
    gray_next,
    gray_prvs,
    prev_pts=None,
    max_points=300,
    min_distance=15,
    block_size=3,
    mask=None,
    quality_level=0.01,
):
    if prev_pts is None:
        # get points
        prev_pts = cv2.goodFeaturesToTrack(
            gray_prvs,
            maxCorners=max_points,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            mask=mask,
        )

    # compute optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        gray_prvs, gray_next, prev_pts, None
    )
    # filter valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx].reshape((-1, 2))
    curr_pts = curr_pts[idx].reshape((-1, 2))
    return curr_pts, prev_pts


class MotionEstimator:
    """
    Estimator of the motion of the camera.

    Uses optical flow to estimate the motion of the camera from frame to frame.
    The optical flow is calculated on a sample of strong points (corners).

    Parameters
    ----------
    max_points : int, optional
        Maximum amount of points sampled.
        More points make the estimation process slower but more precise
    min_distance : int, optional
        Minimum distance between the sample points.
    block_size : int, optional
        Size of an average block when finding the corners. More info in links below.
    transformations_getter : TransformationGetter, optional
        An instance of TransformationGetter. By default [`HomographyTransformationGetter`][norfair.camera_motion.HomographyTransformationGetter]
    draw_flow : bool, optional
        Draws the optical flow on the frame for debugging.
    flow_color : Optional[Tuple[int, int, int]], optional
        Color of the drawing, by default blue.
    quality_level : float, optional
        Parameter characterizing the minimal accepted quality of image corners.

    Examples
    --------
    >>> from norfair import Tracker, Video
    >>> from norfair.camera_motion MotionEstimator
    >>> video = Video("video.mp4")
    >>> tracker = Tracker(...)
    >>> motion_estimator = MotionEstimator()
    >>> for frame in video:
    >>>    detections = get_detections(frame)  # runs detector and returns Detections
    >>>    coord_transformation = motion_estimator.update(frame)
    >>>    tracked_objects = tracker.update(detections, coord_transformations=coord_transformation)

    See Also
    --------
    For more infor on how the points are sampled: [OpenCV.goodFeaturesToTrack](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541)
    """

