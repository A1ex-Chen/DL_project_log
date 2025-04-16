import argparse
import trimesh
import numpy as np
import os
import shutil
import glob
import sys
from multiprocessing import Pool
from functools import partial
sys.path.append(os.getcwd())
from lib.utils.libmesh import check_mesh_contains
from smpl_torch_batch import SMPLModel


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--out_folder', type=str,
                    help='Path to save the outputs.')
parser.add_argument('--ext', type=str, default='obj',
                    help='Extensions for meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--pointcloud_folder', type=str, default='pcl_seq',
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')

parser.add_argument('--points_folder', type=str, default='points_seq',
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=0.5,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                    help='Whether to save truth values as bit array.')








# Export functions





if __name__ == '__main__':
    args = parser.parse_args()
    main(args)