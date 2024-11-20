# Copyright 2024 Microsoft and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class VQDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
            Computed sample x_{t-1} of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.LongTensor










class VQDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler for vector quantized diffusion.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_vec_classes (`int`):
            The number of classes of the vector embeddings of the latent pixels. Includes the class for the masked
            latent pixel.
        num_train_timesteps (`int`, defaults to 100):
            The number of diffusion steps to train the model.
        alpha_cum_start (`float`, defaults to 0.99999):
            The starting cumulative alpha value.
        alpha_cum_end (`float`, defaults to 0.00009):
            The ending cumulative alpha value.
        gamma_cum_start (`float`, defaults to 0.00009):
            The starting cumulative gamma value.
        gamma_cum_end (`float`, defaults to 0.99999):
            The ending cumulative gamma value.
    """

    order = 1

    @register_to_config




