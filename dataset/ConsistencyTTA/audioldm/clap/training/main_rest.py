from inspect import getargs
import logging
import os
import random
from datetime import datetime
import bisect
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import GradScaler
import faulthandler
import pathlib

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, create_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate
from open_clip.utils import dataset_split, get_optimizer






# def updateifNone(a, b):
#     a = b if None else a
#     return a










if __name__ == "__main__":
    main()