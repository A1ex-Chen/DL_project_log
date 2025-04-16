# Copyright 2024 Long Lian, the GLIGEN Authors, and The HuggingFace Team. All rights reserved.
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

# This is a single file implementation of LMD+. See README.md for examples.

import ast
import gc
import inspect
import math
import warnings
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import Attention, GatedSelfAttentionDense
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "longlian/lmd_plus",
        ...     custom_pipeline="llm_grounded_diffusion",
        ...     custom_revision="main",
        ...     variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> # Generate an image described by the prompt and
        >>> # insert objects described by text at the region defined by bounding boxes
        >>> prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
        >>> boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
        >>> phrases = ["a waterfall", "a modern high speed train"]

        >>> images = pipe(
        ...     prompt=prompt,
        ...     phrases=phrases,
        ...     boxes=boxes,
        ...     gligen_scheduled_sampling_beta=0.4,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ...     lmd_guidance_kwargs={}
        ... ).images

        >>> images[0].save("./lmd_plus_generation.jpg")

        >>> # Generate directly from a text prompt and an LLM response
        >>> prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
        >>> phrases, boxes, bg_prompt, neg_prompt = pipe.parse_llm_response(\"""
        [('a waterfall', [71, 105, 148, 258]), ('a modern high speed train', [255, 223, 181, 149])]
        Background prompt: A beautiful forest with fall foliage
        Negative prompt:
        \""")

        >> images = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=neg_prompt,
        ...     phrases=phrases,
        ...     boxes=boxes,
        ...     gligen_scheduled_sampling_beta=0.4,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ...     lmd_guidance_kwargs={}
        ... ).images

        >>> images[0].save("./lmd_plus_generation.jpg")

images[0]

        ```
"""

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# All keys in Stable Diffusion models: [('down', 0, 0, 0), ('down', 0, 1, 0), ('down', 1, 0, 0), ('down', 1, 1, 0), ('down', 2, 0, 0), ('down', 2, 1, 0), ('mid', 0, 0, 0), ('up', 1, 0, 0), ('up', 1, 1, 0), ('up', 1, 2, 0), ('up', 2, 0, 0), ('up', 2, 1, 0), ('up', 2, 2, 0), ('up', 3, 0, 0), ('up', 3, 1, 0), ('up', 3, 2, 0)]
# Note that the first up block is `UpBlock2D` rather than `CrossAttnUpBlock2D` and does not have attention. The last index is always 0 in our case since we have one `BasicTransformerBlock` in each `Transformer2DModel`.
DEFAULT_GUIDANCE_ATTN_KEYS = [
    ("mid", 0, 0, 0),
    ("up", 1, 0, 0),
    ("up", 1, 1, 0),
    ("up", 1, 2, 0),
]




DEFAULT_GUIDANCE_ATTN_KEYS = [convert_attn_keys(key) for key in DEFAULT_GUIDANCE_ATTN_KEYS]




# Adapted from the parent class `AttnProcessor2_0`
class AttnProcessorWithHook(AttnProcessor2_0):



class LLMGroundedDiffusionPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for layout-grounded text-to-image generation using LLM-grounded Diffusion (LMD+): https://arxiv.org/pdf/2305.13655.pdf.

    This model inherits from [`StableDiffusionPipeline`] and aims at implementing the pipeline with minimal modifications. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    This is a simplified implementation that does not perform latent or attention transfer from single object generation to overall generation. The final image is generated directly with attention and adapters control.

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
        requires_safety_checker (bool):
            Whether a safety checker is needed for this pipeline.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    objects_text = "Objects: "
    bg_prompt_text = "Background prompt: "
    bg_prompt_text_no_trailing_space = bg_prompt_text.rstrip()
    neg_prompt_text = "Negative prompt: "
    neg_prompt_text_no_trailing_space = neg_prompt_text.rstrip()



    @classmethod

    @classmethod

    @classmethod









    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)

    @torch.set_grad_enabled(True)

    # Below are methods copied from StableDiffusionPipeline
    # The design choice of not inheriting from StableDiffusionPipeline is discussed here: https://github.com/huggingface/diffusers/pull/5993#issuecomment-1834258517

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.guidance_scale
    @property

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.guidance_rescale
    @property

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.clip_skip
    @property

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.do_classifier_free_guidance
    @property

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.cross_attention_kwargs
    @property

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.num_timesteps
    @property