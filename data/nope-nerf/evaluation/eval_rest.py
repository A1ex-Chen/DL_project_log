import os
from re import L
import sys
import argparse
import time
import logging
import torch

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.common import compute_errors
from model.eval_images import Eval_Images
import model as mdl
import imageio
import numpy as np
import lpips as lpips_lib
from utils_poses.align_traj import align_scale_c2b_use_a2b, align_ate_c2b_use_a2b
from tqdm import tqdm
from model.common import mse2psnr
from torch.utils.tensorboard import SummaryWriter


if __name__=='__main__':
    # Config
    parser = argparse.ArgumentParser(
        description='Extract images.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    eval(cfg)


       