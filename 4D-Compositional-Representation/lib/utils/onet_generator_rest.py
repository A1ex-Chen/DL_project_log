import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from lib.utils import libmcubes
from lib.common import make_3d_grid
from lib.utils.libsimplify import simplify_mesh
from lib.utils.libmise import MISE
import time


class Generator3D(object):
    '''  Generator class for Occupancy Networks.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        padding (float): how much padding should be used for MISE
    '''



