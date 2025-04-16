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


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import UVit2DModel, VQModel
from ...schedulers import AmusedScheduler
from ...utils import replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AmusedInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = AmusedInpaintPipeline.from_pretrained(
        ...     "amused/amused-512", variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "fall mountains"
        >>> input_image = (
        ...     load_image(
        ...         "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg"
        ...     )
        ...     .resize((512, 512))
        ...     .convert("RGB")
        ... )
        >>> mask = (
        ...     load_image(
        ...         "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
        ...     )
        ...     .resize((512, 512))
        ...     .convert("L")
        ... )
        >>> pipe(prompt, input_image, mask).images[0].save("out.png")
        ```
"""


class AmusedInpaintPipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: UVit2DModel
    scheduler: AmusedScheduler

    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    # TODO - when calling self.vqvae.quantize, it uses self.vqvae.quantize.embedding.weight before
    # the forward method of self.vqvae.quantize, so the hook doesn't get called to move the parameter
    # off the meta device. There should be a way to fix this instead of just not offloading it
    _exclude_from_cpu_offload = ["vqvae"]


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)