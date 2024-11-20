import torch
import cv2
import numpy as np
import os.path as osp
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import argparse
from tqdm import tqdm
import time
import os

import sys
sys.path.append('/Labs/Scripts/3DPC/VFMSeg')
sys.path.append('/Labs/Scripts/3DPC/VFMSeg/VFM')
from xmuda.data.utils.augmentation_3d import augment_and_scale_3d
from VFM.sam import build_SAM
from VFM.seem import build_SEEM, call_SEEM

from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from yacs.config import CfgNode as CN



# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


cuda_device_idx = 0
# torch.cuda.set_device(cuda_device_idx)


# Settings
pre_defined_proc_list =  ['A2D2_Resize'] #['SKITTI_A2D2'] #['DAY_NIGHT_Resize'] #['USA_SING_Resize'] # ['A2D2_Resize']#[ 'USA_SING' ,  'DAY_NIGHT'] # 'KITTI', 'A2D2' , 'USA_SING' ,  'DAY_NIGHT'

# Path for VFM ckpt and config files:
vfm_pth='/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/seem_focall_v1.pt'
vfm_cfg='../../VFM/configs/seem/seem_focall_lang.yaml'

#######################
# Config
######################

#  Virtual_SemantiKITTI
vkitti_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/virtual_kitti'
save_new_vkitti_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/virtual_kitti_preprocess_vfm'
pkl_path_VKITTI_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/virtual_kitti_preprocess/preprocess/train.pkl'

save_new_skitti_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/semantic_kitti_preprocess_vfm'
skitti_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/SKITTI/semantic_kitti'
pkl_path_SKITTI_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/train.pkl'
pkl_path_SKITTI_test   = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/test.pkl'
pkl_path_SKITTI_val  = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess/preprocess/val.pkl'
config_path_VKITTI = '/Labs/Scripts/3DPC/VFMSeg/configs/virtual_kitti_semantic_kitti/uda/xmuda_pl.yaml'

#  A2D2_SemanticKITTI
a2d2_orig_preprocess_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/audi/a2d2_preprocess'
save_new_a2d2_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/a2d2_preprocess_vfm'
pkl_path_A2D2_train = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/audi/preprocess/train.pkl'


# nuScene lidarseg
nuScene_orig_data_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes'

save_new_usa_singapore_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/nuscene_preprocess_vfm/usa_singapore'
pkl_path_train_usa_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_usa.pkl'
pkl_path_train_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_singapore.pkl'
pkl_path_test_usa_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_usa.pkl'
pkl_path_test_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_singapore.pkl'
pkl_path_val_singapore_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/val_singapore.pkl'

save_new_day_night_pkl_path = '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/nuscene_preprocess_vfm/day_night'
pkl_path_train_day_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_day.pkl'
pkl_path_train_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/train_night.pkl'
pkl_path_test_day_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_day.pkl'
pkl_path_test_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/test_night.pkl'
pkl_path_val_night_path = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/nuScenes_lidarseg/preprocess/val_night.pkl'

######################

  


















if __name__ == "__main__":

    # test_data()

    print('load seem...')
    SEEM = build_SEEM(vfm_pth,vfm_cfg,cuda_device_idx) #cuda_device_idx
    print('load sam...')
    SAM  = build_SAM(device=cuda_device_idx) #cuda_device_idx
 

    if 'KITTI' in pre_defined_proc_list:
        preprocess_vkitti_train(SEEM,SAM)
        preprocess_skitti(SEEM,SAM,'train')
        preprocess_skitti(SEEM,SAM,'test')
        preprocess_skitti(SEEM,SAM,'val')

    if 'SKITTI_A2D2' in pre_defined_proc_list:
        preprocess_skitti(SEEM,SAM,'train',mapping='A2D2SCN')
        preprocess_skitti(SEEM,SAM,'test',mapping='A2D2SCN')
        preprocess_skitti(SEEM,SAM,'val',mapping='A2D2SCN')

    if 'A2D2' in  pre_defined_proc_list:
        preprcess_a2d2(SEEM,SAM,'train')
    
    if 'USA_SING' in pre_defined_proc_list:
        print('load USA_SING...')
        preprcess_nuScene(SEEM,SAM,'train','USA_SING')
        preprcess_nuScene(SEEM,SAM,'test','USA_SING')
        preprcess_nuScene(SEEM,SAM,'val','USA_SING')

    if 'DAY_NIGHT' in pre_defined_proc_list:
        print('load DAY_NIGHT...')
        preprcess_nuScene(SEEM,SAM,'train','DAY_NIGHT')
        preprcess_nuScene(SEEM,SAM,'test','DAY_NIGHT')
        preprcess_nuScene(SEEM,SAM,'val','DAY_NIGHT')


    if 'A2D2_Resize' in  pre_defined_proc_list:
        preprcess_a2d2(SEEM,SAM,'train',(480,302))

    if 'USA_SING_Resize' in pre_defined_proc_list:
        print('load USA_SING...')
        preprcess_nuScene(SEEM,SAM,'train','USA_SING',(400,225))

    if 'DAY_NIGHT_Resize' in pre_defined_proc_list:
        print('load DAY_NIGHT...')
        preprcess_nuScene(SEEM,SAM,'train','DAY_NIGHT',(400,225))