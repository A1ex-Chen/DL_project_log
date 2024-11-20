# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args

from ..utils import BaseOutput, PushToHubMixin


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


# NOTE: We make this type an enum because it simplifies usage in docs and prevents
# circular imports when used for `_compatibles` within the schedulers module.
# When it's used as a type in pipelines, it really is a Union because the actual
# scheduler instance is passed in.
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15


AysSchedules = {
    "StableDiffusionTimesteps": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "StableDiffusionSigmas": [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0],
    "StableDiffusionXLTimesteps": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "StableDiffusionXLSigmas": [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0],
    "StableDiffusionVideoSigmas": [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.0],
}


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the output of a scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class SchedulerMixin(PushToHubMixin):
    """
    Base class for all schedulers.

    [`SchedulerMixin`] contains common functions shared by all schedulers such as general loading and saving
    functionalities.

    [`ConfigMixin`] takes care of storing the configuration attributes (like `num_train_timesteps`) that are passed to
    the scheduler's `__init__` function, and the attributes can be accessed by `scheduler.config.num_train_timesteps`.

    Class attributes:
        - **_compatibles** (`List[str]`) -- A list of scheduler classes that are compatible with the parent scheduler
          class. Use [`~ConfigMixin.from_config`] to load a different compatible scheduler class (should be overridden
          by parent class).
    """

    config_name = SCHEDULER_CONFIG_NAME
    _compatibles = []
    has_compatibles = True

    @classmethod
    @validate_hf_hub_args


    @property

    @classmethod