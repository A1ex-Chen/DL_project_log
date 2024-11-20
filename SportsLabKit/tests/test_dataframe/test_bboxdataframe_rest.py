import tempfile
import unittest
from collections import namedtuple
from test.support import captured_stdout
from unittest import mock

import numpy as np

from soccertrack import Camera
from soccertrack.dataframe import BBoxDataFrame
from soccertrack.io.file import load_codf
from soccertrack.logger import *
from soccertrack.types import Detection
from soccertrack.utils import get_git_root

csv_path = (
    get_git_root() / "tests" / "assets" / "codf_sample.csv"
)  # already in pitch coordinates
outputs_path = get_git_root() / "tests" / "outputs"

root = (
    get_git_root()
)  # A 2x2 video with 100 frames and goes from red to green to blue to black in 25 frames each

# A 2x2 video with 100 frames and goes from red to green to blue to black in 25 frames each
rgb_video_path = root / "tests" / "assets" / "videos" / "rgb_video.avi"


class TestBBoxDataFrame(unittest.TestCase):





