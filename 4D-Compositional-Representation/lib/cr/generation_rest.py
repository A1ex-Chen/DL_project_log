import torch
import os
import numpy as np
from lib.utils.io import save_mesh
from trimesh.exchange.export import export_mesh
import time
from lib.utils.onet_generator import Generator3D as Generator3DONet


class Generator3D(object):
    '''  Generator class for Occupancy Networks 4D.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        device (device): pytorch device
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        n_time_steps (int): number of time steps to generate
        only_ent_time_points (bool): whether to only generate end points
    '''











