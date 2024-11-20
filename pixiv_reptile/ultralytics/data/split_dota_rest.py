# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools
from glob import glob
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils.checks import check_requirements

check_requirements("shapely")
from shapely.geometry import Polygon


















if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")