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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...image_processor import PipelineImageInput
from ...loaders import IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...models.unets.unet_motion_model import MotionAdapter
from ...schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..free_init_utils import FreeInitMixin
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import AnimateDiffPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        >>> from diffusers.utils import export_to_gif

        >>> adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        >>> pipe = AnimateDiffPipeline.from_pretrained("frankjoshua/toonyou_beta6", motion_adapter=adapter)
        >>> pipe.scheduler = DDIMScheduler(beta_schedule="linear", steps_offset=1, clip_sample=False)
        >>> output = pipe(prompt="A corgi walking in the park")
        >>> frames = output.frames[0]
        >>> export_to_gif(frames, "animation.gif")
        ```
"""


class AnimateDiffPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    FreeInitMixin,
):
    r"""
    Pipeline for text-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A [`UNet2DConditionModel`] used to create a UNetMotionModel to denoise the encoded video latents.
        motion_adapter ([`MotionAdapter`]):
            A [`MotionAdapter`] to be used in combination with `unet` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt with num_images_per_prompt -> num_videos_per_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds

    # Copied from diffusers.pipelines.text_to_video_synthesis/pipeline_text_to_video_synth.TextToVideoSDPipeline.decode_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs

    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents

    @property

    @property

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property

    @property

    @property

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)