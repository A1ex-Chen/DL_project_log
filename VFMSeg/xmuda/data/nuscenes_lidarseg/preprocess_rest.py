import os
import os.path as osp
import numpy as np
import pickle

import sys
sys.path.append('/Labs/Scripts/3DPC/exp_xmuda_journal')

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

from xmuda.data.nuscenes_lidarseg.projection import map_pointcloud_to_image
from xmuda.data.nuscenes_lidarseg import splits






if __name__ == '__main__':
    root_dir = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes'
    out_dir  = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg'
    # root_dir = '/datasets_master/nuscenes'
    # out_dir = '/datasets_local/datasets_mjaritz/nuscenes_lidarseg_preprocess_debug/'
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')

    # SemanticKITTI/nuScenes-lidarseg (to evaluate against LiDAR transfer baseline)
    # preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, subset_name='all')