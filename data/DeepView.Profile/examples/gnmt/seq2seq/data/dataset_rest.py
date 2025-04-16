import logging
from operator import itemgetter

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import seq2seq.data.config as config
from seq2seq.data.sampler import BucketingSampler
from seq2seq.data.sampler import DistributedSampler
from seq2seq.data.sampler import ShardingSampler
from seq2seq.data.sampler import StaticDistributedSampler





    if parallel:
        return parallel_collate
    else:
        return single_collate


class TextDataset(Dataset):









class ParallelDataset(TextDataset):





class LazyParallelDataset(TextDataset):



