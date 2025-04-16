import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator

from xmuda.data.utils.visualize import save_2D_segmentations
from torchvision.utils import save_image as Tensor2Img
from PIL import Image 

from VFM.seem import build_SEEM, call_SEEM


cuda_device_idx = 1

torch.cuda.set_device(cuda_device_idx)



