import threading
from queue import Queue
from time import sleep

import cv2
import numpy as np


class VideoReader:
    """Pythonic wrapper around OpenCV's VideoCapture().

    This class provides a convenient way to access and manipulate video files using OpenCV's VideoCapture object. It implements several convenient methods and properties to make it easy to work with video files, including slicing and indexing into the video file, iteration through the video frames, and more.

    Args:
        filename (str): The path to the video file.
        threaded (bool): Whether to run the video reading in a separate thread.
        queue_size (int): The size of the queue for storing video frames.

    Properties:
        frame_width (int): The width of the video frames.
        frame_height (int): The height of the video frames.
        frame_channels (int): The number of channels in the video frames.
        frame_rate (float): The frame rate of the video.
        frame_shape (tuple): The shape of the video frames (height, width, channels).
        number_of_frames (int): The total number of frames in the video.
        fourcc (int): The fourcc code of the video.
        current_frame_pos (int): The current position of the video frame.

    Methods:
        read(frame_number=None): Read the next frame or a specified frame from the video.
        close(): Close the video file.
    """















    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property