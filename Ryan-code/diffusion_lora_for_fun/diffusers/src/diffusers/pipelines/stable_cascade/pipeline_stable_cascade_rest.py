# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import is_torch_version, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
        ... ).to("cuda")
        >>> gen_pipe = StableCascadeDecoderPipeline.from_pretrain(
        ...     "stabilityai/stable-cascade", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        >>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)
        ```
"""


class StableCascadeDecoderPipeline(DiffusionPipeline):
    """
    Pipeline for generating images from the Stable Cascade model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`CLIPTokenizer`):
            The CLIP tokenizer.
        text_encoder (`CLIPTextModel`):
            The CLIP text encoder.
        decoder ([`StableCascadeUNet`]):
            The Stable Cascade decoder unet.
        vqgan ([`PaellaVQModel`]):
            The VQGAN model.
        scheduler ([`DDPMWuerstchenScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        latent_dim_scale (float, `optional`, defaults to 10.67):
            Multiplier to determine the VQ latent space size from the image embeddings. If the image embeddings are
            height=24 and width=24, the VQ latent shape needs to be height=int(24*10.67)=256 and
            width=int(24*10.67)=256 in order to match the training conditions.
    """

    unet_name = "decoder"
    text_encoder_name = "text_encoder"
    model_cpu_offload_seq = "text_encoder->decoder->vqgan"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_pooled",
        "negative_prompt_embeds",
        "image_embeddings",
    ]





    @property

    @property

    @property

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)