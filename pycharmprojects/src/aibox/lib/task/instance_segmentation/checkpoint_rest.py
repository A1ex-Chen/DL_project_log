from dataclasses import dataclass

import torch

from .algorithm import Algorithm
from .model import Model
from ...checkpoint import Checkpoint as Base


@dataclass
class Checkpoint(Base):

    @staticmethod

    @staticmethod