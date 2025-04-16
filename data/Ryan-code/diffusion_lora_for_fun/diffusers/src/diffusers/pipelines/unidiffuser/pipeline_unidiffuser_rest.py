import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    GPT2Tokenizer,
)

from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ...utils.outputs import BaseOutput
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .modeling_text_decoder import UniDiffuserTextDecoder
from .modeling_uvit import UniDiffuserModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# New BaseOutput child class for joint image-text output
@dataclass
class ImageTextPipelineOutput(BaseOutput):
    """
    Output class for joint image-text pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        text (`List[str]` or `List[List[str]]`)
            List of generated text strings of length `batch_size` or a list of list of strings whose outer list has
            length `batch_size`.
    """

    images: Optional[Union[List[PIL.Image.Image], np.ndarray]]
    text: Optional[Union[List[str], List[List[str]]]]


class UniDiffuserPipeline(DiffusionPipeline):
    r"""
    Pipeline for a bimodal image-text model which supports unconditional text and image generation, text-conditioned
    image generation, image-conditioned text generation, and joint image-text generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations. This
            is part of the UniDiffuser image representation along with the CLIP vision encoding.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        image_encoder ([`CLIPVisionModel`]):
            A [`~transformers.CLIPVisionModel`] to encode images as part of its image representation along with the VAE
            latent representation.
        image_processor ([`CLIPImageProcessor`]):
            [`~transformers.CLIPImageProcessor`] to preprocess an image before CLIP encoding it with `image_encoder`.
        clip_tokenizer ([`CLIPTokenizer`]):
             A [`~transformers.CLIPTokenizer`] to tokenize the prompt before encoding it with `text_encoder`.
        text_decoder ([`UniDiffuserTextDecoder`]):
            Frozen text decoder. This is a GPT-style model which is used to generate text from the UniDiffuser
            embedding.
        text_tokenizer ([`GPT2Tokenizer`]):
            A [`~transformers.GPT2Tokenizer`] to decode text for text generation; used along with the `text_decoder`.
        unet ([`UniDiffuserModel`]):
            A [U-ViT](https://github.com/baofff/U-ViT) model with UNNet-style skip connections between transformer
            layers to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image and/or text latents. The
            original UniDiffuser paper uses the [`DPMSolverMultistepScheduler`] scheduler.
    """

    # TODO: support for moving submodules for components with enable_model_cpu_offload
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae->text_decoder"


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs


    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_tiling

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_tiling

    # Functions to manually set the mode







    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt with self.tokenizer->self.clip_tokenizer

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix.StableDiffusionInstructPix2PixPipeline.prepare_image_latents
    # Add num_prompts_per_image argument, sample from autoencoder moment distribution



    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    # Rename prepare_latents -> prepare_image_vae_latents and add num_prompts_per_image argument.










    @torch.no_grad()