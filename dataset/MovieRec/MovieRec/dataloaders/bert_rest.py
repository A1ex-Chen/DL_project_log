from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):

    @classmethod









class BertTrainDataset(data_utils.Dataset):






class BertEvalDataset(data_utils.Dataset):


