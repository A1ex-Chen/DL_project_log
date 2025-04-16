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
sys.path.append('/Labs/Scripts/3DPC/exp_VFMSeg')
sys.path.append('/Labs/Scripts/3DPC/exp_VFMSeg/VFM')

cuda_device_idx = 1
torch.cuda.set_device(cuda_device_idx)

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate

from VFM.seem import build_SEEM, call_SEEM












if __name__ == '__main__':
    main()