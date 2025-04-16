import logging

import torch
from torch.utils.data.sampler import Sampler

from seq2seq.utils import get_rank
from seq2seq.utils import get_world_size


class DistributedSampler(Sampler):








class ShardingSampler(DistributedSampler):



class BucketingSampler(DistributedSampler):



class StaticDistributedSampler(Sampler):

