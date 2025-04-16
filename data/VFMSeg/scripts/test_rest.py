#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import numpy as np

import torch
import torch.nn.functional as F

import sys
sys.path.append('/Labs/Scripts/3DPC/VFMSeg')
sys.path.append('/Labs/Scripts/3DPC/VFMSeg/VFM')
torch.cuda.set_device(0)

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate










if __name__ == '__main__':
    main()