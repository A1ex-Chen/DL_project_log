# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """


