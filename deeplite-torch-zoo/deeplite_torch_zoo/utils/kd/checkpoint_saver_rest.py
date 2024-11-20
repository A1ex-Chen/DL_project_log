""" Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os

import torch

from timm.utils.model import unwrap_model, get_state_dict
from deeplite_torch_zoo.utils import LOGGER


class CheckpointSaver:  # don't save optimizer state dict, since it forces specific repo sturcture




