# Copyright 2023 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    add_noise_common,
)


@flax.struct.dataclass
class DPMSolverMultistepSchedulerState:
    common: CommonSchedulerState
    alpha_t: jnp.ndarray
    sigma_t: jnp.ndarray
    lambda_t: jnp.ndarray

    # setable values
    init_noise_sigma: jnp.ndarray
    timesteps: jnp.ndarray
    num_inference_steps: Optional[int] = None

    # running values
    model_outputs: Optional[jnp.ndarray] = None
    lower_order_nums: Optional[jnp.int32] = None
    prev_timestep: Optional[jnp.int32] = None
    cur_sample: Optional[jnp.ndarray] = None

    @classmethod


@dataclass
class FlaxDPMSolverMultistepSchedulerOutput(FlaxSchedulerOutput):
    state: DPMSolverMultistepSchedulerState


class FlaxDPMSolverMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver for diffusion ODEs with
    the convergence order guarantee. Empirically, sampling by DPM-Solver with only 20 steps can generate high-quality
    samples, and it can generate quite good samples even in only 10 steps.

    For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095

    Currently, we support the multistep DPM-Solver for both noise prediction models and data prediction models. We
    recommend to use `solver_order=2` for guided sampling, and `solver_order=3` for unconditional sampling.

    We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
    diffusion models, you can set both `algorithm_type="dpmsolver++"` and `thresholding=True` to use the dynamic
    thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models (such as
    stable-diffusion).

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        solver_order (`int`, default `2`):
            the order of DPM-Solver; can be `1` or `2` or `3`. We recommend to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the data / `x0`. One of `epsilon`, `sample`,
            or `v-prediction`.
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            For pixel-space diffusion models, you can set both `algorithm_type=dpmsolver++` and `thresholding=True` to
            use the dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion
            models (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487).
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++`.
        algorithm_type (`str`, default `dpmsolver++`):
            the algorithm type for the solver. Either `dpmsolver` or `dpmsolver++`. The `dpmsolver` type implements the
            algorithms in https://arxiv.org/abs/2206.00927, and the `dpmsolver++` type implements the algorithms in
            https://arxiv.org/abs/2211.01095. We recommend to use `dpmsolver++` with `solver_order=2` for guided
            sampling (e.g. stable-diffusion).
        solver_type (`str`, default `midpoint`):
            the solver type for the second-order solver. Either `midpoint` or `heun`. The solver type slightly affects
            the sample quality, especially for small number of steps. We empirically find that `midpoint` solvers are
            slightly better, so we recommend to use the `midpoint` type.
        lower_order_final (`bool`, default `True`):
            whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
            find this trick can stabilize the sampling of DPM-Solver for steps < 15, especially for steps <= 10.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            the `dtype` used for params and computation.
    """

    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    dtype: jnp.dtype

    @property

    @register_to_config













            step_2_output = step_2(state)
            step_3_output = step_3(state)

            if self.config.solver_order == 2:
                return step_2_output
            elif self.config.lower_order_final and len(state.timesteps) < 15:
                return jax.lax.select(
                    state.lower_order_nums < 2,
                    step_2_output,
                    jax.lax.select(
                        step_index == len(state.timesteps) - 2,
                        step_2_output,
                        step_3_output,
                    ),
                )
            else:
                return jax.lax.select(
                    state.lower_order_nums < 2,
                    step_2_output,
                    step_3_output,
                )

        step_1_output = step_1(state)
        step_23_output = step_23(state)

        if self.config.solver_order == 1:
            prev_sample = step_1_output

        elif self.config.lower_order_final and len(state.timesteps) < 15:
            prev_sample = jax.lax.select(
                state.lower_order_nums < 1,
                step_1_output,
                jax.lax.select(
                    step_index == len(state.timesteps) - 1,
                    step_1_output,
                    step_23_output,
                ),
            )

        else:
            prev_sample = jax.lax.select(
                state.lower_order_nums < 1,
                step_1_output,
                step_23_output,
            )

        state = state.replace(
            lower_order_nums=jnp.minimum(state.lower_order_nums + 1, self.config.solver_order),
        )

        if not return_dict:
            return (prev_sample, state)

        return FlaxDPMSolverMultistepSchedulerOutput(prev_sample=prev_sample, state=state)

    def scale_model_input(
        self, state: DPMSolverMultistepSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
        return sample

    def add_noise(
        self,
        state: DPMSolverMultistepSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        return add_noise_common(state.common, original_samples, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps