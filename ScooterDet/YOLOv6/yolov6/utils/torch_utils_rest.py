#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import time
from contextlib import contextmanager
from copy import deepcopy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from yolov6.utils.events import LOGGER

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


@contextmanager









