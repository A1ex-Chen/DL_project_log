import logging
import os
import random
from datetime import datetime
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import time

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

from clap_module import create_model_and_transforms, trace_model, create_model
from training.data import get_data
from training.params import parse_args
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.scheduler import cosine_lr
from training.lp_train import train_one_epoch, evaluate
from clap_module.utils import get_tar_path_from_dataset_name, dataset_split, get_optimizer
from clap_module.utils import load_p, load_class_label
from clap_module.linear_probe import LinearProbe






# def updateifNone(a, b):
#     a = b if None else a
#     return a











if __name__ == "__main__":
    main()