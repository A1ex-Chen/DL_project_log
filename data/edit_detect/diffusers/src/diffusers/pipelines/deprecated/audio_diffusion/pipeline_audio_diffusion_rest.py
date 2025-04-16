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


from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ....models import AutoencoderKL, UNet2DConditionModel
from ....schedulers import DDIMScheduler, DDPMScheduler
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import AudioPipelineOutput, BaseOutput, DiffusionPipeline, ImagePipelineOutput
from .mel import Mel


class AudioDiffusionPipeline(DiffusionPipeline):
    """
    Pipeline for audio diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vqae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        mel ([`Mel`]):
            Transform audio into a spectrogram.
        scheduler ([`DDIMScheduler`] or [`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`] or [`DDPMScheduler`].
    """

    _optional_components = ["vqvae"]



    @torch.no_grad()

    @torch.no_grad()

    @staticmethod