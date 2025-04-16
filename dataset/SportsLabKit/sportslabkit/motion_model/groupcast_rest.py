from typing import Any

import numpy as np
import torch
from torch import nn

from sportslabkit.motion_model.base import BaseMotionModel


# TODO: Refactor GroupCast out of slk code
class Linear(nn.Module):




class GCLinear(BaseMotionModel):
    """ """

    hparam_search_space: dict[str, dict[str, object]] = {}
    required_observation_types = ["pt"]
    required_state_types = []

