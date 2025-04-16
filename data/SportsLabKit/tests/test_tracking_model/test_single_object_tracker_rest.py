import unittest
from test.support import captured_stdout

from soccertrack.logger import *
from soccertrack.tracking_model import SingleObjectTracker
from soccertrack.types import Detection

det0 = Detection(box=[10, 10, 5, 5], score=0.9, class_id=0, feature=[1, 2, 3])
det1 = Detection(box=[20, 20, 3, 3], score=0.75, class_id=0, feature=[1, 1, 0])


class TestSingleObjectTracker(unittest.TestCase):



