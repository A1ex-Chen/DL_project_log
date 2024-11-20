from typing import Callable, Dict, List, Optional, Union

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
        >>> from diffusers import AutoPipelineForText2Image
        >>> import torch

        >>> pipe = AutoPipelineForText2Image.from_pretrained(
        ...     "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        >>> generator = torch.Generator(device="cpu").manual_seed(0)
        >>> image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
        ```

"""




class Kandinsky3Pipeline(DiffusionPipeline, LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->unet->movq"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_attention_mask",
        "attention_mask",
    ]



    @torch.no_grad()



    @property

    @property

    @property

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)