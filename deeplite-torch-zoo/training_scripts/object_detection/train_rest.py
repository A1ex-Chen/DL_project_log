# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from addict import Dict
import yaml

# reduce cpu usage for dataloading
# must set before importing torch, or set in terminal
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import lr_scheduler
from cv2 import setNumThreads

# reduce cpu usage for dataloading in distributed setting
setNumThreads(0)

from deeplite_torch_zoo import get_eval_function, get_dataloaders, get_model

from deeplite_torch_zoo.src.object_detection.yolo.losses import YOLOv5Loss
from deeplite_torch_zoo.utils import strip_optimizer, LOGGER, TQDM_BAR_FORMAT, colorstr, increment_path, \
    init_seeds, one_cycle, print_args, yaml_save, check_img_size, \
    EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))










if __name__ == '__main__':
    opt = parse_opt()
    main(opt)