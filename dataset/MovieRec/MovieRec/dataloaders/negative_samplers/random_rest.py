from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
