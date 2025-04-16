import inspect
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import deprecate, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import SemanticStableDiffusionPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SemanticStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with latent editing.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).

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
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents

    @torch.no_grad()