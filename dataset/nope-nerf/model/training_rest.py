import os
import torch
import logging
from model.losses import Loss
import numpy as np
from PIL import Image
import imageio
from torch.nn import functional as F
from model.common import (
    get_tensor_values, 
     arange_pixels,  project_to_cam, transform_to_world,
)
logger_py = logging.getLogger(__name__)
class Trainer(object):


    
        
    