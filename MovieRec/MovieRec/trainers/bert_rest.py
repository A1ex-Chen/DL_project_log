from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTTrainer(AbstractTrainer):

    @classmethod




