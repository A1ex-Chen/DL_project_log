# Copyright 2024 Salesforce.com, inc.
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
from typing import List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPTokenizer

from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from ...schedulers import PNDMScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..blip_diffusion.blip_image_processing import BlipImageProcessor
from ..blip_diffusion.modeling_blip2 import Blip2QFormerModel
from ..blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers.pipelines import BlipDiffusionControlNetPipeline
        >>> from diffusers.utils import load_image
        >>> from controlnet_aux import CannyDetector
        >>> import torch

        >>> blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
        ...     "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> style_subject = "flower"
        >>> tgt_subject = "teapot"
        >>> text_prompt = "on a marble table"

        >>> cldm_cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/kettle.jpg"
        ... ).resize((512, 512))
        >>> canny = CannyDetector()
        >>> cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
        >>> style_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/flower.jpg"
        ... )
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 50
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


        >>> output = blip_diffusion_pipe(
        ...     text_prompt,
        ...     style_image,
        ...     cldm_cond_image,
        ...     style_subject,
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


class BlipDiffusionControlNetPipeline(DiffusionPipeline):
    """
    Pipeline for Canny Edge based Controlled subject-driven generation using Blip Diffusion.

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
        controlnet ([`ControlNetModel`]):
            ControlNet model to get the conditioning image embedding.
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """

    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"



    # from the original Blip Diffusion code, speciefies the target subject and augments the prompt by repeating it

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents


    # Adapted from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)