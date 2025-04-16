# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function
from torch.nn import functional as F
from torch.cuda.amp import autocast

from detectron2.utils import comm, env
from detectron2.layers.wrappers import BatchNorm2d

class AllReduce(Function):
    @staticmethod

    @staticmethod

class MergedSyncBatchNorm(BatchNorm2d):
    """
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """


            

    # @float_function