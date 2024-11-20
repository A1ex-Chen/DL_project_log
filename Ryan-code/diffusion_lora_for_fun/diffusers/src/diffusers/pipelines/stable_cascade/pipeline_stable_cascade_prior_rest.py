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

from dataclasses import dataclass
from math import ceil
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import BaseOutput, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEFAULT_STAGE_C_TIMESTEPS = list(np.linspace(1.0, 2 / 3, 20)) + list(np.linspace(2 / 3, 0.0, 11))[1:]

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadePriorPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
        ... ).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        ```
"""


@dataclass
class StableCascadePriorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        image_embeddings (`torch.Tensor` or `np.ndarray`)
            Prior image embeddings for text prompt
        prompt_embeds (`torch.Tensor`):
            Text embeddings for the prompt.
        negative_prompt_embeds (`torch.Tensor`):
            Text embeddings for the negative prompt.
    """

    image_embeddings: Union[torch.Tensor, np.ndarray]
    prompt_embeds: Union[torch.Tensor, np.ndarray]
    prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]
    negative_prompt_embeds: Union[torch.Tensor, np.ndarray]
    negative_prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]


class StableCascadePriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Stable Cascade.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`StableCascadeUNet`]):
            The Stable Cascade prior to approximate the image embedding from the text and/or image embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`DDPMWuerstchenScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        resolution_multiple ('float', *optional*, defaults to 42.67):
            Default resolution for multiple images generated.
    """

    unet_name = "prior"
    text_encoder_name = "text_encoder"
    model_cpu_offload_seq = "image_encoder->text_encoder->prior"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "text_encoder_hidden_states", "negative_prompt_embeds"]






    @property

    @property

    @property


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)