import unittest

import functools as ft
import itertools as it

from apex import amp
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT


class TestBasicCasts(unittest.TestCase):









class TestBannedMethods(unittest.TestCase):





class TestTensorCasts(unittest.TestCase):








class TestDisabledCasts(unittest.TestCase):


    # TODO: maybe more tests on disabled casting?

if __name__ == '__main__':
    unittest.main()