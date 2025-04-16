import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from lib.data.core import Field
from lib.common import random_crop_occ


class IndexField(Field):
    ''' Basic index field.'''
    # def load(self, model_path, idx, category):




class PointsSubseqField(Field):
    ''' Points subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        all_steps (bool): whether to return all time steps
        fixed_time_step (int): if and which fixed time step to use
        unpackbits (bool): whether to unpack bits
        scale_type (str, optional): Specifies the type of transformation to apply to the point cloud:
        ``'cr'`` | ``'oflow'``. ``'cr'``: transform the point cloud to align with the output,
        ``'oflow'``: scale the point cloud w.r.t. the first point cloud of the sequence
        spatial_completion (bool): whether to remove some points for 4D spatial completion experiment
    '''









class PointCloudSubseqField(Field):
    ''' Point cloud subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        only_end_points (bool): whether to only return end points
        scale_type (str, optional): Specifies the type of transformation to apply to the input point cloud:
        ``'cr'`` | ``'oflow'``. ``'cr'``: transform the point cloud the original scale and location of SMPL model,
        ``'oflow'``: scale the point cloud w.r.t. the first point cloud of the sequence
    '''






