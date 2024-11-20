# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""






if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    heatmap = Heatmap(classes_names)