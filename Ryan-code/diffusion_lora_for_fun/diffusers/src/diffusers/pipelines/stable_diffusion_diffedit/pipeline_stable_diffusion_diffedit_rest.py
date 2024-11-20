# Copyright 2024 DiffEdit Authors and Pix2Pix Zero Authors and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import DDIMInverseScheduler, KarrasDiffusionSchedulers
from ...utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from ..stable_diffusion import StableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class DiffEditInversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.Tensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `num_timesteps * batch_size` or numpy array of shape `(num_timesteps,
            batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the
            diffusion pipeline.
    """

    latents: torch.Tensor
    images: Union[List[PIL.Image.Image], np.ndarray]


EXAMPLE_DOC_STRING = """

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionDiffEditPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

        >>> init_image = download_image(img_url).resize((768, 768))

        >>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.enable_model_cpu_offload()

        >>> mask_prompt = "A bowl of fruits"
        >>> prompt = "A bowl of pears"

        >>> mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
        >>> image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
        >>> image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]
        ```
"""

EXAMPLE_INVERT_DOC_STRING = """
        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionDiffEditPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

        >>> init_image = download_image(img_url).resize((768, 768))

        >>> pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.enable_model_cpu_offload()

        >>> prompt = "A bowl of fruits"

        >>> inverted_latents = pipe.invert(image=init_image, prompt=prompt).latents
        ```
"""






# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess




class StableDiffusionDiffEditPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, LoraLoaderMixin
):
    r"""
    <Tip warning={true}>

    This is an experimental feature!

    </Tip>

    Pipeline for text-guided image inpainting using Stable Diffusion and DiffEdit.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading and saving methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        inverse_scheduler ([`DDIMInverseScheduler`]):
            A scheduler to be used in combination with `unet` to fill in the unmasked part of the input latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "inverse_scheduler"]
    _exclude_from_cpu_offload = ["safety_checker"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents





    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)