""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2019, Ross Wightman
"""

from functools import partial

import torch

from timm.data.dataset import IterableImageDataset
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader, _worker_init

from deeplite_torch_zoo.src.classification.augmentations.augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

