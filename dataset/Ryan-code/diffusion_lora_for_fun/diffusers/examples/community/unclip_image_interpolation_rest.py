import inspect
from typing import List, Optional, Union

import PIL.Image
import torch
from torch.nn import functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DiffusionPipeline,
    ImagePipelineOutput,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class UnCLIPImageInterpolationPipeline(DiffusionPipeline):
    """
    Pipeline to generate variations from an input image using unCLIP

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. unCLIP Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution unet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution unet. Used in the last step of the super resolution diffusion process.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process. Just a modified DDPMScheduler.
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process. Just a modified DDPMScheduler.

    """

    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline.__init__

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline._encode_prompt

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline._encode_image

    @torch.no_grad()