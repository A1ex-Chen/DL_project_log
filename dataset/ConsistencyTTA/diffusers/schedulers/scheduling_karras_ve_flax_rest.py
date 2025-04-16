# Copyright 2023 NVIDIA and The HuggingFace Team. All rights reserved.
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

import flax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils_flax import FlaxSchedulerMixin


@flax.struct.dataclass
class KarrasVeSchedulerState:
    # setable values
    num_inference_steps: Optional[int] = None
    timesteps: Optional[jnp.ndarray] = None
    schedule: Optional[jnp.ndarray] = None  # sigma(t_i)

    @classmethod


@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        derivative (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Derivative of predicted original image sample (x_0).
        state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.
    """

    prev_sample: jnp.ndarray
    derivative: jnp.ndarray
    state: KarrasVeSchedulerState


class FlaxKarrasVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details on the parameters, see the original paper's Appendix E.: "Elucidating the Design Space of
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. The grid search values used to find the
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    Args:
        sigma_min (`float`): minimum noise magnitude
        sigma_max (`float`): maximum noise magnitude
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].
    """

    @property

    @register_to_config





