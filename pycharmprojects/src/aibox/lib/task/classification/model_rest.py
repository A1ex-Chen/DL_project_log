from typing import Union, Tuple, Dict

import torch
from torch.nn import functional as F

from .algorithm import Algorithm
from .preprocessor import Preprocessor
from ...extension.data_parallel import Bunch
from ...extension.functional import normalize_means_stds
from ...model import Model as Base


class Model(Base):

