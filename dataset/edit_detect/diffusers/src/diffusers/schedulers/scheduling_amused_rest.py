import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin






@dataclass
class AmusedSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None


class AmusedScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: torch.Tensor

    @register_to_config


