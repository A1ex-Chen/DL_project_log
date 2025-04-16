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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from packaging import version
from transformers import CLIPImageProcessor, XLMRobertaTokenizer

from diffusers.utils import is_accelerate_available, is_accelerate_version

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...loaders import TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import PIL_INTERPOLATION, deprecate, logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from . import AltDiffusionPipelineOutput, RobertaSeriesModelWithTransformation


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from diffusers import AltDiffusionImg2ImgPipeline

        >>> device = "cuda"
        >>> model_id_or_path = "BAAI/AltDiffusion-m9"
        >>> pipe = AltDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        >>> pipe = pipe.to(device)

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        >>> response = requests.get(url)
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> init_image = init_image.resize((768, 512))

        >>> # "A fantasy landscape, trending on artstation"
        >>> prompt = "幻想风景, artstation"

        >>> images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
        >>> images[0].save("幻想风景.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline with Stable->Alt, CLIPTextModel->RobertaSeriesModelWithTransformation, CLIPTokenizer->XLMRobertaTokenizer, AltDiffusionSafetyChecker->StableDiffusionSafetyChecker
class AltDiffusionImg2ImgPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    r"""
    Pipeline for text-guided image to image generation using Alt Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`RobertaSeriesModelWithTransformation`]):
            Frozen text-encoder. Alt Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.RobertaSeriesModelWithTransformation),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`XLMRobertaTokenizer`):
            Tokenizer of class
            [XLMRobertaTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.XLMRobertaTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]




    @property








    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)