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

import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import deprecate, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.preprocess


class StableDiffusionLatentUpscalePipeline(DiffusionPipeline, StableDiffusionMixin, FromSingleFileMixin):
    r"""
    Pipeline for upscaling Stable Diffusion output image resolution by a factor of 2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A [`EulerDiscreteScheduler`] to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"



    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.prepare_latents

    @torch.no_grad()