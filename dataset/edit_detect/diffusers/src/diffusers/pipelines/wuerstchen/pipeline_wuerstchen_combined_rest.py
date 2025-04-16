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

from ...schedulers import DDPMWuerstchenScheduler
from ...utils import deprecate, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .modeling_paella_vq_model import PaellaVQModel
from .modeling_wuerstchen_diffnext import WuerstchenDiffNeXt
from .modeling_wuerstchen_prior import WuerstchenPrior
from .pipeline_wuerstchen import WuerstchenDecoderPipeline
from .pipeline_wuerstchen_prior import WuerstchenPriorPipeline


TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusions import WuerstchenCombinedPipeline

        >>> pipe = WuerstchenCombinedPipeline.from_pretrained("warp-ai/Wuerstchen", torch_dtype=torch.float16).to(
        ...     "cuda"
        ... )
        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> images = pipe(prompt=prompt)
        ```
"""


class WuerstchenCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for text-to-image generation using Wuerstchen

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer (`CLIPTokenizer`):
            The decoder tokenizer to be used for text inputs.
        text_encoder (`CLIPTextModel`):
            The decoder text encoder to be used for text inputs.
        decoder (`WuerstchenDiffNeXt`):
            The decoder model to be used for decoder image generation pipeline.
        scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for decoder image generation pipeline.
        vqgan (`PaellaVQModel`):
            The VQGAN model to be used for decoder image generation pipeline.
        prior_tokenizer (`CLIPTokenizer`):
            The prior tokenizer to be used for text inputs.
        prior_text_encoder (`CLIPTextModel`):
            The prior text encoder to be used for text inputs.
        prior_prior (`WuerstchenPrior`):
            The prior model to be used for prior pipeline.
        prior_scheduler (`DDPMWuerstchenScheduler`):
            The scheduler to be used for prior pipeline.
    """

    _load_connected_pipes = True







    @torch.no_grad()
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)