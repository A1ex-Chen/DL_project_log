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

# DISCLAIMER: check https://arxiv.org/abs/2302.04867 and https://github.com/wl-zhao/UniPC for more info
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput



    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class UniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    UniPC is a training-free framework designed for the fast sampling of diffusion models, which consists of a
    corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders. UniPC is
    by desinged model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can
    also be applied to both noise prediction model and data prediction model. The corrector UniC can be also applied
    after any off-the-shelf solvers to increase the order of accuracy.

    For more details, see the original paper: https://arxiv.org/abs/2302.04867

    Currently, we support the multistep UniPC for both noise prediction models and data prediction models. We recommend
    to use `solver_order=2` for guided sampling, and `solver_order=3` for unconditional sampling.

    We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). For pixel-space
    diffusion models, you can set both `predict_x0=True` and `thresholding=True` to use the dynamic thresholding. Note
    that the thresholding method is unsuitable for latent-space diffusion models (such as stable-diffusion).

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

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
            the order of UniPC, also the p in UniPC-p; can be any positive integer. Note that the effective order of
            accuracy is `solver_order + 1` due to the UniC. We recommend to use `solver_order=2` for guided sampling,
            and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            For pixel-space diffusion models, you can set both `predict_x0=True` and `thresholding=True` to use the
            dynamic thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models
            (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487).
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True` and `predict_x0=True`.
        predict_x0 (`bool`, default `True`):
            whether to use the updating algrithm on the predicted x0. See https://arxiv.org/abs/2211.01095 for details
        solver_type (`str`, default `bh2`):
            the solver type of UniPC. We recommend use `bh1` for unconditional sampling when steps < 10, and use `bh2`
            otherwise.
        lower_order_final (`bool`, default `True`):
            whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. We empirically
            find this trick can stabilize the sampling of DPM-Solver for steps < 15, especially for steps <= 10.
        disable_corrector (`list`, default `[]`):
            decide which step to disable the corrector. For large guidance scale, the misalignment between the
            `epsilon_theta(x_t, c)`and `epsilon_theta(x_t^c, c)` might influence the convergence. This can be mitigated
            by disable the corrector at the first few steps (e.g., disable_corrector=[0])
        solver_p (`SchedulerMixin`, default `None`):
            can be any other scheduler. If specified, the algorithm will become solver_p + UniC.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config


    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample






