# Copyright 2024 Salesforce.com, inc.
# Copyright 2024 The HuggingFace Team. All rights reserved.#
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
from typing import List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import PNDMScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .blip_image_processing import BlipImageProcessor
from .modeling_blip2 import Blip2QFormerModel
from .modeling_ctx_clip import ContextCLIPTextModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers.pipelines import BlipDiffusionPipeline
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
        ...     "Salesforce/blipdiffusion", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> cond_subject = "dog"
        >>> tgt_subject = "dog"
        >>> text_prompt_input = "swimming underwater"

        >>> cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
        ... )
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 25
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


        >>> output = blip_diffusion_pipe(
        ...     text_prompt_input,
        ...     cond_image,
        ...     cond_subject,
        ...     tgt_subject,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=num_inference_steps,
        ...     neg_prompt=negative_prompt,
        ...     height=512,
        ...     width=512,
        ... ).images
        >>> output[0].save("image.png")
        ```
"""


class BlipDiffusionPipeline(DiffusionPipeline):
    """
    Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the text encoder
        text_encoder ([`ContextCLIPTextModel`]):
            Text encoder to encode the text prompt
        vae ([`AutoencoderKL`]):
            VAE model to map the latents to the image
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        scheduler ([`PNDMScheduler`]):
             A scheduler to be used in combination with `unet` to generate image latents.
        qformer ([`Blip2QFormerModel`]):
            QFormer model to get multi-modal embeddings from the text and image.
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """

    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"



    # from the original Blip Diffusion code, speciefies the target subject and augments the prompt by repeating it

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)