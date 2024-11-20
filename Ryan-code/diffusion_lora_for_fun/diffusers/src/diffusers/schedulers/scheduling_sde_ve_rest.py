# Copyright 2024 Google Brain and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import SchedulerMixin, SchedulerOutput


@dataclass
class SdeVeOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample` over previous timesteps.
    """

    prev_sample: torch.Tensor
    prev_sample_mean: torch.Tensor


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVeScheduler` is a variance exploding stochastic differential equation (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        snr (`float`, defaults to 0.15):
            A coefficient weighting the step from the `model_output` sample (from the network) to the random noise.
        sigma_min (`float`, defaults to 0.01):
            The initial noise scale for the sigma sequence in the sampling procedure. The minimum sigma should mirror
            the distribution of the data.
        sigma_max (`float`, defaults to 1348.0):
            The maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`, defaults to 1e-5):
            The end value of sampling where timesteps decrease progressively from 1 to epsilon.
        correct_steps (`int`, defaults to 1):
            The number of correction steps performed on a produced sample.
    """

    order = 1

    @register_to_config







