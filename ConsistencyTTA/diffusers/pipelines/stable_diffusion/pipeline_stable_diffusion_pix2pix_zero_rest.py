# Copyright 2023 Pix2Pix Zero Authors and The HuggingFace Team. All rights reserved.
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
import PIL
import torch
import torch.nn.functional as F
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from ...loaders import TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention_processor import Attention
from ...schedulers import DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
from ...schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from ...utils import (
    PIL_INTERPOLATION,
    BaseOutput,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Pix2PixInversionPipelineOutput(BaseOutput, TextualInversionLoaderMixin):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    latents: torch.FloatTensor
    images: Union[List[PIL.Image.Image], np.ndarray]


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import requests
        >>> import torch

        >>> from diffusers import DDIMScheduler, StableDiffusionPix2PixZeroPipeline


        >>> def download(embedding_url, local_filepath):
        ...     r = requests.get(embedding_url)
        ...     with open(local_filepath, "wb") as f:
        ...         f.write(r.content)


        >>> model_ckpt = "CompVis/stable-diffusion-v1-4"
        >>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.to("cuda")

        >>> prompt = "a high resolution painting of a cat in the style of van gough"
        >>> source_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/cat.pt"
        >>> target_emb_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/dog.pt"

        >>> for url in [source_emb_url, target_emb_url]:
        ...     download(url, url.split("/")[-1])

        >>> src_embeds = torch.load(source_emb_url.split("/")[-1])
        >>> target_embeds = torch.load(target_emb_url.split("/")[-1])
        >>> images = pipeline(
        ...     prompt,
        ...     source_embeds=src_embeds,
        ...     target_embeds=target_embeds,
        ...     num_inference_steps=50,
        ...     cross_attention_guidance_amount=0.15,
        ... ).images

        >>> images[0].save("edited_image_dog.png")
        ```
"""

EXAMPLE_INVERT_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from transformers import BlipForConditionalGeneration, BlipProcessor
        >>> from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

        >>> import requests
        >>> from PIL import Image

        >>> captioner_id = "Salesforce/blip-image-captioning-base"
        >>> processor = BlipProcessor.from_pretrained(captioner_id)
        >>> model = BlipForConditionalGeneration.from_pretrained(
        ...     captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ... )

        >>> sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
        >>> pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        ...     sd_model_ckpt,
        ...     caption_generator=model,
        ...     caption_processor=processor,
        ...     torch_dtype=torch.float16,
        ...     safety_checker=None,
        ... )

        >>> pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.enable_model_cpu_offload()

        >>> img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"

        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))
        >>> # generate caption
        >>> caption = pipeline.generate_caption(raw_image)

        >>> # "a photography of a cat with flowers and dai dai daie - daie - daie kasaii"
        >>> inv_latents = pipeline.invert(caption, image=raw_image).latents
        >>> # we need to generate source and target embeds

        >>> source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]

        >>> target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]

        >>> source_embeds = pipeline.get_embeds(source_prompts)
        >>> target_embeds = pipeline.get_embeds(target_prompts)
        >>> # the latents can then be used to edit a real image
        >>> # when using Stable Diffusion 2 or other models that use v-prediction
        >>> # set `cross_attention_guidance_amount` to 0.01 or less to avoid input latent gradient explosion

        >>> image = pipeline(
        ...     caption,
        ...     source_embeds=source_embeds,
        ...     target_embeds=target_embeds,
        ...     num_inference_steps=50,
        ...     cross_attention_guidance_amount=0.15,
        ...     generator=generator,
        ...     latents=inv_latents,
        ...     negative_prompt=caption,
        ... ).images[0]
        >>> image.save("edited_image.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess




class Pix2PixZeroL2Loss:



class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""




class StableDiffusionPix2PixZeroPipeline(DiffusionPipeline):
    r"""
    Pipeline for pixel-levl image editing using Pix2Pix Zero. Based on Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`], or [`DDPMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        requires_safety_checker (bool):
            Whether the pipeline requires a safety checker. We recommend setting it to True if you're using the
            pipeline publicly.
    """
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "caption_generator",
        "caption_processor",
        "inverse_scheduler",
    ]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload


    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs


    #  Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents

    @torch.no_grad()


    @torch.no_grad()





    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_INVERT_DOC_STRING)