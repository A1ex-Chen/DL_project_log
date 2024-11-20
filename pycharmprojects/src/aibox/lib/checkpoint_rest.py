from dataclasses import dataclass

import torch
from torch.optim.optimizer import Optimizer

from .model import Model


@dataclass
class Checkpoint:
    epoch: int
    model: Model
    optimizer: Optimizer

    @staticmethod

    @staticmethod