from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod

