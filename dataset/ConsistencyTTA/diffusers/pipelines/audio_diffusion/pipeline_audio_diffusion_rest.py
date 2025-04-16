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


from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import DDIMScheduler, DDPMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, BaseOutput, DiffusionPipeline, ImagePipelineOutput
from .mel import Mel


class AudioDiffusionPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqae ([`AutoencoderKL`]): Variational AutoEncoder for Latent Audio Diffusion or None
        unet ([`UNet2DConditionModel`]): UNET model
        mel ([`Mel`]): transform audio <-> spectrogram
        scheduler ([`DDIMScheduler` or `DDPMScheduler`]): de-noising scheduler
    """

    _optional_components = ["vqvae"]




    @torch.no_grad()

    @torch.no_grad()

    @staticmethod