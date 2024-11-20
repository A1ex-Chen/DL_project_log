import torch
import torch.nn as nn

from deeplite_torch_zoo.utils.profiler import profile_macs
from deeplite_torch_zoo.src.registries import ZERO_COST_SCORES


@ZERO_COST_SCORES.register('macs')