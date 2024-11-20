from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Dict, Union, Tuple

from torch import Tensor

from .algorithm import Algorithm
from ... import config
from ...config import REQUIRED


@dataclass
class Config(config.Config):

    algorithm_name: Algorithm.Name = REQUIRED

    pretrained: bool = True
    num_frozen_levels: int = 2

    eval_center_crop_ratio: float = 1


    @staticmethod