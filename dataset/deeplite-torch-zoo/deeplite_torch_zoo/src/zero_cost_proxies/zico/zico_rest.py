'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import numpy as np

import torch
from torch import nn

from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES
from deeplite_torch_zoo.src.zero_cost_proxies.utils import aggregate_statistic






@ZERO_COST_SCORES.register('zico')