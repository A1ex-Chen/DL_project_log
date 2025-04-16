from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple

from torch import Tensor

from .algorithm import Algorithm
from ... import config
from ...config import REQUIRED


@dataclass
class Config(config.Config):

    algorithm_name: Algorithm.Name = REQUIRED


    @staticmethod