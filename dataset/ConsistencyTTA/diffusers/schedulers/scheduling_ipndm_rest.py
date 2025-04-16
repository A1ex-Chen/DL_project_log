# Copyright 2023 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class IPNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Improved Pseudo numerical methods for diffusion models (iPNDM) ported from @crowsonkb's amazing k-diffusion
    [library](https://github.com/crowsonkb/v-diffusion-pytorch/blob/987f8985e38208345c1959b0ea767a625831cc9b/diffusion/sampling.py#L296)

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2202.09778

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
    """

    order = 1

    @register_to_config




