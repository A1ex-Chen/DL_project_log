# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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
    A fourth-order Improved Pseudo Linear Multistep scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
    """

    order = 1

    @register_to_config

    @property

    @property

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index


    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index



