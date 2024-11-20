# Copyright (c) Facebook, Inc. and its affiliates.

import torch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Cannot import tensorboard. Will log to txt files only.")
    SummaryWriter = None

from utils.dist import is_primary


class Logger(object):
