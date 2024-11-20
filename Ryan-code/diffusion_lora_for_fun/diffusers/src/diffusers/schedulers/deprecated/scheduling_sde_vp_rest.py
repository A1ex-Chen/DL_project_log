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
from typing import Union

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.torch_utils import randn_tensor
from ..scheduling_utils import SchedulerMixin


class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVpScheduler` is a variance preserving stochastic differential equation (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 2000):
            The number of diffusion steps to train the model.
        beta_min (`int`, defaults to 0.1):
        beta_max (`int`, defaults to 20):
        sampling_eps (`int`, defaults to 1e-3):
            The end value of sampling where timesteps decrease progressively from 1 to epsilon.
    """

    order = 1

    @register_to_config


