# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import List, Optional, Tuple, Union

import torch

from ...models import UNet2DModel, VQModel
from ...schedulers import DDIMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class LDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is to be used in combination with `unet` to denoise the encoded image latents.
    """


    @torch.no_grad()