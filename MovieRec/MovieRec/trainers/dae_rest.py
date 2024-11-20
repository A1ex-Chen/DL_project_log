from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAETrainer(AbstractTrainer):

    @classmethod




