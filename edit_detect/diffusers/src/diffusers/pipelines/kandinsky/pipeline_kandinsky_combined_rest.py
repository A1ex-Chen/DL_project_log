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
from typing import Callable, List, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    XLMRobertaTokenizer,
)

from ...models import PriorTransformer, UNet2DConditionModel, VQModel
from ...schedulers import DDIMScheduler, DDPMScheduler, UnCLIPScheduler
from ...utils import (
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from .pipeline_kandinsky import KandinskyPipeline
from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
from .pipeline_kandinsky_prior import KandinskyPriorPipeline
from .text_encoder import MultilingualCLIP


TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        ```
"""

IMAGE2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForImage2Image
        import torch
        import requests
        from io import BytesIO
        from PIL import Image
        import os

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.thumbnail((768, 768))

        image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
        ```
"""

INPAINT_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        from diffusers import AutoPipelineForInpainting
        from diffusers.utils import load_image
        import torch
        import numpy as np

        pipe = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        original_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
        )

        mask = np.zeros((768, 768), dtype=np.float32)
        # Let's mask out an area above the cat's head
        mask[:250, 250:-250] = 1

        image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
        ```
"""


class KandinskyCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "text_encoder->unet->movq->prior_prior->prior_image_encoder->prior_text_encoder"
    _exclude_from_cpu_offload = ["prior_prior"]






    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)


class KandinskyImg2ImgCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->" "text_encoder->unet->movq"
    _exclude_from_cpu_offload = ["prior_prior"]






    @torch.no_grad()
    @replace_example_docstring(IMAGE2IMAGE_EXAMPLE_DOC_STRING)


class KandinskyInpaintCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    _load_connected_pipes = True
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->text_encoder->unet->movq"
    _exclude_from_cpu_offload = ["prior_prior"]






    @torch.no_grad()
    @replace_example_docstring(INPAINT_EXAMPLE_DOC_STRING)