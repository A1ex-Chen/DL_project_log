# Copyright 2024 ParaDiGMS authors and The HuggingFace Team. All rights reserved.
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
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ....image_processor import VaeImageProcessor
from ....loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ....models import AutoencoderKL, UNet2DConditionModel
from ....models.lora import adjust_lora_scale_text_encoder
from ....schedulers import KarrasDiffusionSchedulers
from ....utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from ...stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from ...stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DDPMParallelScheduler
        >>> from diffusers import StableDiffusionParadigmsPipeline

        >>> scheduler = DDPMParallelScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        >>> pipe = StableDiffusionParadigmsPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", scheduler=scheduler, torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> ngpu, batch_per_device = torch.cuda.device_count(), 5
        >>> pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, parallel=ngpu * batch_per_device, num_inference_steps=1000).images[0]
        ```
"""


class StableDiffusionParadigmsPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using a parallelized version of Stable Diffusion.

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

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)