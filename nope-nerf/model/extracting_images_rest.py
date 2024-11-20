import os
import torch
from collections import defaultdict
from model.common import (
    get_tensor_values, arange_pixels
)
from tqdm import tqdm
import logging
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
import imageio
class Extract_Images(object):

