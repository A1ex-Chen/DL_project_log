import os
import sys
import argparse
import torch
import cv2
import numpy as np
import imageio

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
import model as mdl

 
                                                                
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess.'
    )
    parser.add_argument('config', type=str,default='configs/preprocess.yaml', help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    if cfg['dataloading']['crop_size'] !=0:
        folder_name= 'dpt_' + str(cfg['dataloading']['crop_size'])
    else:
        folder_name = 'dpt'
    depth_save_dir = os.path.join(cfg['dataloading']['path'], cfg['dataloading']['scene'][0], folder_name)
    dpt_depth(cfg, depth_save_dir)