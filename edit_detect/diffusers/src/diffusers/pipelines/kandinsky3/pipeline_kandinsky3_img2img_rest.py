import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image
import torch
from transformers import T5EncoderModel, T5Tokenizer

from ...loaders import LoraLoaderMixin
from ...models import Kandinsky3UNet, VQModel
from ...schedulers import DDPMScheduler
from ...utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import AutoPipelineForImage2Image
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> pipe = AutoPipelineForImage2Image.from_pretrained(
        ...     "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A painting of the inside of a subway train with tiny raccoons."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png"
        ... )

        >>> generator = torch.Generator(device="cpu").manual_seed(0)
        >>> image = pipe(prompt, image=image, strength=0.75, num_inference_steps=25, generator=generator).images[0]
        ```
"""






class Kandinsky3Img2ImgPipeline(DiffusionPipeline, LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->movq->unet->movq"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_attention_mask",
        "attention_mask",
    ]




    @torch.no_grad()


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs


    @property

    @property

    @property

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)