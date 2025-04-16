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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    """
    Multistep and onestep sampling for consistency models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 40):
            The number of diffusion steps to train the model.
        sigma_min (`float`, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. Defaults to 0.002 from the original implementation.
        sigma_max (`float`, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. Defaults to 80.0 from the original implementation.
        sigma_data (`float`, defaults to 0.5):
            The standard deviation of the data distribution from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 0.5 from the original implementation.
        s_noise (`float`, defaults to 1.0):
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011]. Defaults to 1.0 from the original implementation.
        rho (`float`, defaults to 7.0):
            The parameter for calculating the Karras sigma schedule from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 7.0 from the original implementation.
        clip_denoised (`bool`, defaults to `True`):
            Whether to clip the denoised outputs to `(-1, 1)`.
        timesteps (`List` or `np.ndarray` or `torch.Tensor`, *optional*):
            An explicit timestep schedule that can be optionally specified. The timesteps are expected to be in
            increasing order.
    """

    order = 1

    @register_to_config

    @property

    @property

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index




    # Modified _convert_to_karras implementation that takes in ramp as argument



    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index


    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
