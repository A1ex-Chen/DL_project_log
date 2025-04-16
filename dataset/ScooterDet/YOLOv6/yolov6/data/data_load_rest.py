#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import os
import torch.distributed as dist
from torch.utils.data import dataloader, distributed

from .datasets import TrainValDataset
from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import torch_distributed_zero_first




class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """





class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

