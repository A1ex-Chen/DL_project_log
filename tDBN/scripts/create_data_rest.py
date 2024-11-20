import copy
import pathlib
import pickle

import fire
import numpy as np
from skimage import io as imgio


import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'tDBN'))


from tDBN.core import box_np_ops
from tDBN.core.point_cloud.point_cloud_ops import bound_points_jit
from tDBN.kitti import kitti_common as kitti
from tDBN.utils.progress_bar import list_bar as prog_bar
"""
Note: tqdm has problem in my system(win10), so use my progress bar
try:
    from tqdm import tqdm as prog_bar
except ImportError:
    from tDBN.utils.progress_bar import progress_bar_iter as prog_bar
"""














if __name__ == '__main__':
    fire.Fire()