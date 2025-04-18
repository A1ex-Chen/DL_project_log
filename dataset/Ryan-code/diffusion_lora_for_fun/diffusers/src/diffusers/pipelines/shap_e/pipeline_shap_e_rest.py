# Copyright 2024 Open AI and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...models import PriorTransformer
from ...schedulers import HeunDiscreteScheduler
from ...utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .renderer import ShapERenderer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline
        >>> from diffusers.utils import export_to_gif

        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        >>> repo = "openai/shap-e"
        >>> pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
        >>> pipe = pipe.to(device)

        >>> guidance_scale = 15.0
        >>> prompt = "a shark"

        >>> images = pipe(
        ...     prompt,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=64,
        ...     frame_size=256,
        ... ).images

        >>> gif_path = export_to_gif(images[0], "shark_3d.gif")
        ```
"""


@dataclass
class ShapEPipelineOutput(BaseOutput):
    """
    Output class for [`ShapEPipeline`] and [`ShapEImg2ImgPipeline`].

    Args:
        images (`torch.Tensor`)
            A list of images for 3D rendering.
    """

    images: Union[List[List[PIL.Image.Image]], List[List[np.ndarray]]]


class ShapEPipeline(DiffusionPipeline):
    """
    Pipeline for generating latent representation of a 3D asset and rendering with the NeRF method.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_encoder ([`~transformers.CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer ([`~transformers.CLIPTokenizer`]):
             A `CLIPTokenizer` to tokenize text.
        scheduler ([`HeunDiscreteScheduler`]):
            A scheduler to be used in combination with the `prior` model to generate image embedding.
        shap_e_renderer ([`ShapERenderer`]):
            Shap-E renderer projects the generated latents into parameters of a MLP to create 3D objects with the NeRF
            rendering method.
    """

    model_cpu_offload_seq = "text_encoder->prior"
    _exclude_from_cpu_offload = ["shap_e_renderer"]


    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)