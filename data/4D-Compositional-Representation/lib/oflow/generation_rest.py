import torch
import numpy as np
import trimesh
import os
from lib.common import load_and_scale_mesh
from lib.utils.io import save_mesh
import time
from lib.utils.onet_generator_oflow import Generator3D as Generator3DONet
import pandas as pd


class Generator3D(Generator3DONet):
    ''' OFlow Generator object class.

    It provides methods to extract final meshes from the OFlow representation.

    Args:
        model (nn.Module): OFlow model
        device (device): PyTorch device
        points_batch_size (int): batch size for evaluation points to extract
            the shape at t=0
        threshold (float): threshold value for the Occupancy Networks-based
            shape representation at t=0
        refinement_step (int): number of refinement step for MISE
        padding (float): padding value for MISE
        sample (bool): whether to sample from prior for latent code z
        simplify_nfaces (int): number of faces the mesh should be simplified to
        n_time_steps (int): number of time steps which should be extracted
        mesh_color (bool): whether to save the meshes with color
            encoding
        only_end_time_points (bool): whether to only generate first and last
            mesh
        interpolate (bool): whether to use the velocity field to interpolate
            between start and end mesh
        fix_z (bool): whether to hold latent shape code fixed
        fix_zt (bool): whether to hold latent motion code fixed
    '''

















