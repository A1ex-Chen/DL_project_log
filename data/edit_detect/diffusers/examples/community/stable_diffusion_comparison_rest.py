from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


pipe1_model_id = "CompVis/stable-diffusion-v1-1"
pipe2_model_id = "CompVis/stable-diffusion-v1-2"
pipe3_model_id = "CompVis/stable-diffusion-v1-3"
pipe4_model_id = "CompVis/stable-diffusion-v1-4"


class StableDiffusionComparisonPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for parallel comparison of Stable Diffusion v1-v4
    This pipeline inherits from DiffusionPipeline and depends on the use of an Auth Token for
    downloading pre-trained checkpoints from Hugging Face Hub.
    If using Hugging Face Hub, pass the Model ID for Stable Diffusion v1.4 as the previous 3 checkpoints will be loaded
    automatically.
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
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionMegaSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """


    @property

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()