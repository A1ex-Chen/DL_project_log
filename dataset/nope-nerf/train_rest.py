import os
import sys
import logging
import time
import argparse

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl
from utils_poses.comp_ate import compute_ATE, compute_rpe
from model.common import backup,  mse2psnr
from utils_poses.align_traj import align_ate_c2b_use_a2b

if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Training of nope-nerf model'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = dl.load_config(args.config, 'configs/default.yaml')
    # backup model
    backup(cfg['training']['out_dir'], args.config)
    train(cfg=cfg)
    