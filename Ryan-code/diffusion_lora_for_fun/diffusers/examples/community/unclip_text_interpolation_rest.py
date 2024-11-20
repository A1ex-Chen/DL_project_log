import inspect
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from diffusers import (
    DiffusionPipeline,
    ImagePipelineOutput,
    PriorTransformer,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class UnCLIPTextInterpolationPipeline(DiffusionPipeline):
    """
    Pipeline for prompt-to-prompt interpolation on CLIP text embeddings and using the UnCLIP / Dall-E to decode them to images.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution unet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution unet. Used in the last step of the super resolution diffusion process.
        prior_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the prior denoising process. Just a modified DDPMScheduler.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process. Just a modified DDPMScheduler.
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process. Just a modified DDPMScheduler.

    """

    prior: PriorTransformer
    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    prior_scheduler: UnCLIPScheduler
    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.__init__

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline._encode_prompt

    @torch.no_grad()