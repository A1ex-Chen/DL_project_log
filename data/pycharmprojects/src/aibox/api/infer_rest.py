import argparse
import base64
import glob
import os
import random
import sys
import time
from ast import literal_eval
from io import BytesIO
from typing import Tuple, List, Optional, Union, Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageStat
from matplotlib import cm
from aibox.lib.bbox import BBox
from aibox.lib.checkpoint import Checkpoint
from aibox.lib.task import Task
# from aibox.lib.task.instance_segmentation.palette import Palette
# from aibox.transforms.functional import to_pil_image
from tqdm import tqdm




if __name__ == '__main__':

    main()