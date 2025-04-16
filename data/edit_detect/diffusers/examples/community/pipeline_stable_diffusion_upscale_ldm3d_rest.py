# Copyright 2024 The Intel Labs Team Authors and the HuggingFace Team. All rights reserved.
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
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import DiffusionPipeline
from diffusers.image_processor import PipelineDepthInput, PipelineImageInput, VaeImageProcessorLDM3D
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion_ldm3d.pipeline_stable_diffusion_ldm3d import LDM3DPipelineOutput
from diffusers.schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> from diffusers import StableDiffusionUpscaleLDM3DPipeline
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> import requests

        >>> pipe = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("Intel/ldm3d-sr")
        >>> pipe = pipe.to("cuda")
        >>> rgb_path = "https://huggingface.co/Intel/ldm3d-sr/resolve/main/lemons_ldm3d_rgb.jpg"
        >>> depth_path = "https://huggingface.co/Intel/ldm3d-sr/resolve/main/lemons_ldm3d_depth.png"
        >>> low_res_rgb = Image.open(BytesIO(requests.get(rgb_path).content)).convert("RGB")
        >>> low_res_depth = Image.open(BytesIO(requests.get(depth_path).content)).convert("L")
        >>> output = pipe(
        ...     prompt="high quality high resolution uhd 4k image",
        ...     rgb=low_res_rgb,
        ...     depth=low_res_depth,
        ...     num_inference_steps=50,
        ...     target_res=[1024, 1024],
        ... )
        >>> rgb_image, depth_image = output.rgb, output.depth
        >>> rgb_image[0].save("hr_ldm3d_rgb.jpg")
        >>> depth_image[0].save("hr_ldm3d_depth.png")
        ```
"""


class StableDiffusionUpscaleLDM3DPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image and 3D generation using LDM3D.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
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
        low_res_scheduler ([`SchedulerMixin`]):
            A scheduler used to add initial noise to the low resolution conditioning image. It must be an instance of
            [`DDPMScheduler`].
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d.StableDiffusionLDM3DPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d.StableDiffusionLDM3DPipeline.encode_prompt


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs



    # def upcast_vae(self):
    #     dtype = self.vae.dtype
    #     self.vae.to(dtype=torch.float32)
    #     use_torch_2_0_or_xformers = isinstance(
    #         self.vae.decoder.mid_block.attentions[0].processor,
    #         (
    #             AttnProcessor2_0,
    #             XFormersAttnProcessor,
    #             LoRAXFormersAttnProcessor,
    #             LoRAAttnProcessor2_0,
    #         ),
    #     )
    #     # if xformers or torch_2_0 is used attention block does not need
    #     # to be in float32 which can save lots of memory
    #     if use_torch_2_0_or_xformers:
    #         self.vae.post_quant_conv.to(dtype)
    #         self.vae.decoder.conv_in.to(dtype)
    #         self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()