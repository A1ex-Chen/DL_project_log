from time import time
import argparse
import logging
import os
import json
from tqdm.auto import tqdm
import math
import numpy as np
import wandb

import torch
import datasets
import transformers
from transformers import SchedulerType

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
logger = get_logger(__name__)

import diffusers
from tools.t2a_dataset import get_dataloaders
from tools.build_pretrained import build_pretrained_models
from tools.train_utils import \
    train_one_epoch, eval_model, log_results, get_optimizer_and_scheduler
from tools.torch_tools import seed_all
from models import AudioGDM, AudioLCM, AudioLCM_FTVAE

TARGET_LENGTH = 1024






if __name__ == "__main__":
    main()