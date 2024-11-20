# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch.cuda import amp
from tqdm import tqdm

from deeplite_torch_zoo.utils import (LOGGER, GenericLogger, ModelEMA, colorstr, init_seeds, print_args,
                                      yaml_save, increment_path, WorkingDirectory,
                                      select_device, smart_DDP, smart_optimizer,
                                      smartCrossEntropyLoss, torch_distributed_zero_first)

from deeplite_torch_zoo import get_model, get_dataloaders, get_eval_function
from deeplite_torch_zoo.utils.kd import KDTeacher, compute_kd_loss


ROOT = Path.cwd()

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))








if __name__ == "__main__":
    opt = parse_opt()
    main(opt)