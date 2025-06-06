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

import numpy as np
import torch
import tqdm

from ...models.unets.unet_1d import UNet1DModel
from ...pipelines import DiffusionPipeline
from ...utils.dummy_pt_objects import DDPMScheduler
from ...utils.torch_utils import randn_tensor


class ValueGuidedRLPipeline(DiffusionPipeline):
    r"""
    Pipeline for value-guided sampling from a diffusion model trained to predict sequences of states.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        value_function ([`UNet1DModel`]):
            A specialized UNet for fine-tuning trajectories base on reward.
        unet ([`UNet1DModel`]):
            UNet architecture to denoise the encoded trajectories.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded trajectories. Default for this
            application is [`DDPMScheduler`].
        env ():
            An environment following the OpenAI gym API to act in. For now only Hopper has pretrained models.
    """






