# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
import os
import logging

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .io import get_md5


class Checkpointer(object):
    """Checkpoint the model and relevant states.

    Supported features:
    1. Resume optimizer and scheduler
    2. Automatically deal with DataParallel, DistributedDataParallel
    3. Resume last saved checkpoint

    """









class CheckpointerV2(Checkpointer):
    """Support max_to_keep like tf.Saver"""





