import os
import time
from typing import List, Optional, Union

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()
import numpy as np
from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn

from norfair import metrics

from .utils import get_terminal_size


class Video:
    """
    Class that provides a simple and pythonic way to interact with video.

    It returns regular OpenCV frames which enables the usage of the huge number of tools OpenCV provides to modify images.

    Parameters
    ----------
    camera : Optional[int], optional
        An integer representing the device id of the camera to be used as the video source.

        Webcams tend to have an id of `0`. Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
    input_path : Optional[str], optional
        A string consisting of the path to the video file to be used as the video source.

        Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
    output_path : str, optional
        The path to the output video to be generated.
        Can be a folder were the file will be created or a full path with a file name.
    output_fps : Optional[float], optional
        The frames per second at which to encode the output video file.

        If not provided it is set to be equal to the input video source's fps.
        This argument is useful when using live video cameras as a video source,
        where the user may know the input fps,
        but where the frames are being fed to the output video at a rate that is lower than the video source's fps,
        due to the latency added by the detector.
    label : str, optional
        Label to add to the progress bar that appears when processing the current video.
    output_fourcc : Optional[str], optional
        OpenCV encoding for output video file.
        By default we use `mp4v` for `.mp4` and `XVID` for `.avi`. This is a combination that works on most systems but
        it results in larger files. To get smaller files use `avc1` or `H264` if available.
        Notice that some fourcc are not compatible with some extensions.
    output_extension : str, optional
        File extension used for the output video. Ignored if `output_path` is not a folder.

    Examples
    --------
    >>> video = Video(input_path="video.mp4")
    >>> for frame in video:
    >>>     # << Your modifications to the frame would go here >>
    >>>     video.write(frame)
    """


    # This is a generator, note the yield keyword below.








class VideoFromFrames:


