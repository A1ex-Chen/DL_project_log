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

import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...models import StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler
from ...utils import is_torch_version, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from ..wuerstchen.modeling_paella_vq_model import PaellaVQModel
from .pipeline_stable_cascade import StableCascadeDecoderPipeline
from .pipeline_stable_cascade_prior import StableCascadePriorPipeline


TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableCascadeCombinedPipeline

        >>> pipe = StableCascadeCombinedPipeline.from_pretrained(
        ...     "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()
        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> images = pipe(prompt=prompt)
        ```
"""


class StableCascadeCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Stable Cascade.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`CLIPTokenizer`):
            The decoder tokenizer to be used for text inputs.
        text_encoder (`CLIPTextModel`):
            The decoder text encoder to be used for text inputs.
        decoder (`StableCascadeUNet`):
            The decoder model to be used for decoder image generation pipeline.
        scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for decoder image generation pipeline.
        vqgan (`PaellaVQModel`):
            The VQGAN model to be used for decoder image generation pipeline.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        prior_prior (`StableCascadeUNet`):
            The prior model to be used for prior pipeline.
        prior_scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for prior pipeline.
    """

    _load_connected_pipes = True
    _optional_components = ["prior_feature_extractor", "prior_image_encoder"]







    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)