import numpy as np

from norfair.camera_motion import TranslationTransformation
from norfair.utils import warn_once


class FixedCamera:
    """
    Class used to stabilize video based on the camera motion.

    Starts with a larger frame, where the original frame is drawn on top of a black background.
    As the camera moves, the smaller frame moves in the opposite direction, stabilizing the objects in it.

    Useful for debugging or demoing the camera motion.
    ![Example GIF](../../videos/camera_stabilization.gif)

    !!! Warning
        This only works with [`TranslationTransformation`][norfair.camera_motion.TranslationTransformation],
        using [`HomographyTransformation`][norfair.camera_motion.HomographyTransformation] will result in
        unexpected behaviour.

    !!! Warning
        If using other drawers, always apply this one last. Using other drawers on the scaled up frame will not work as expected.

    !!! Note
        Sometimes the camera moves so far from the original point that the result won't fit in the scaled-up frame.
        In this case, a warning will be logged and the frames will be cropped to avoid errors.

    Parameters
    ----------
    scale : float, optional
        The resulting video will have a resolution of `scale * (H, W)` where HxW is the resolution of the original video.
        Use a bigger scale if the camera is moving too much.
    attenuation : float, optional
        Controls how fast the older frames fade to black.

    Examples
    --------
    >>> # setup
    >>> tracker = Tracker("frobenious", 100)
    >>> motion_estimator = MotionEstimator()
    >>> video = Video(input_path="video.mp4")
    >>> fixed_camera = FixedCamera()
    >>> # process video
    >>> for frame in video:
    >>>     coord_transformations = motion_estimator.update(frame)
    >>>     detections = get_detections(frame)
    >>>     tracked_objects = tracker.update(detections, coord_transformations)
    >>>     draw_tracked_objects(frame, tracked_objects)  # fixed_camera should always be the last drawer
    >>>     bigger_frame = fixed_camera.adjust_frame(frame, coord_transformations)
    >>>     video.write(bigger_frame)
    """

