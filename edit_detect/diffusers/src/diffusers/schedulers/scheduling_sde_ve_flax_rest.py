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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax import random

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left


@flax.struct.dataclass
class ScoreSdeVeSchedulerState:
    # setable values
    timesteps: Optional[jnp.ndarray] = None
    discrete_sigmas: Optional[jnp.ndarray] = None
    sigmas: Optional[jnp.ndarray] = None

    @classmethod


@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    """
    Output class for the ScoreSdeVeScheduler's step function output.

    Args:
        state (`ScoreSdeVeSchedulerState`):
        prev_sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
    """

    state: ScoreSdeVeSchedulerState
    prev_sample: jnp.ndarray
    prev_sample_mean: Optional[jnp.ndarray] = None


class FlaxScoreSdeVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        snr (`float`):
            coefficient weighting the step from the model_output sample (from the network) to the random noise.
        sigma_min (`float`):
                initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
                distribution of the data.
        sigma_max (`float`): maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`): the end value of sampling, where timesteps decrease progressively from 1 to
        epsilon.
        correct_steps (`int`): number of correction steps performed on a produced sample.
    """

    @property

    @register_to_config





