# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import logging

import torch
import torch.nn as nn

from utils.model_loading import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):


