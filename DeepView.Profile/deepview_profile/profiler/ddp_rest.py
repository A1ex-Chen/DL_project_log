from scipy.stats import gaussian_kde
import numpy as np
import os
import logging
from deepview_profile.pytorch_profiler_log_reader import (
    get_first_last_step,
    get_bucket_sizes,
    get_ddp_forward_backward_times,
)
import time
from torch.profiler import profile, schedule, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess

logger = logging.getLogger(__name__)

FILENAME = "pytorch_profiler.json"
RANK = 0
WORLD_SIZE = 1
DEFAULT_BUCKET_SIZE = 25















