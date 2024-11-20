import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import LCMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import numpy as np

        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_interpolate")
        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        >>> prompts = ["A cat", "A dog", "A horse"]
        >>> num_inference_steps = 4
        >>> num_interpolation_steps = 24
        >>> seed = 1337

        >>> torch.manual_seed(seed)
        >>> np.random.seed(seed)

        >>> images = pipe(
                prompt=prompts,
                height=512,
                width=512,
                num_inference_steps=num_inference_steps,
                num_interpolation_steps=num_interpolation_steps,
                guidance_scale=8.0,
                embedding_interpolation_type="lerp",
                latent_interpolation_type="slerp",
                process_batch_size=4, # Make it higher or lower based on your GPU memory
                generator=torch.Generator(seed),
            )

        >>> # Save the images as a video
        >>> import imageio
        >>> from PIL import Image

        >>> def pil_to_video(images: List[Image.Image], filename: str, fps: int = 60) -> None:
                frames = [np.array(image) for image in images]
                with imageio.get_writer(filename, fps=fps) as video_writer:
                    for frame in frames:
                        video_writer.append_data(frame)

        >>> pil_to_video(images, "lcm_interpolate.mp4", fps=24)
        ```
"""






class LatentConsistencyModelWalkPipeline(
    DiffusionPipeline, StableDiffusionMixin, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using a latent consistency model.

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            supports [`LCMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
        requires_safety_checker (`bool`, *optional*, defaults to `True`):
            Whether the pipeline requires a safety checker component.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "denoised", "prompt_embeds", "w_embedding"]


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Currently StableDiffusionPipeline.check_inputs with negative prompt stuff removed

    @torch.no_grad()

    @torch.no_grad()

    @property

    @property

    @property

    @property

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)