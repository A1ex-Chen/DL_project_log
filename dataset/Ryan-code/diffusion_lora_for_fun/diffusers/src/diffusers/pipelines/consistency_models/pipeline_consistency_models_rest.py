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

from typing import Callable, List, Optional, Union

import torch

from ...models import UNet2DModel
from ...schedulers import CMStochasticIterativeScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch

        >>> from diffusers import ConsistencyModelPipeline

        >>> device = "cuda"
        >>> # Load the cd_imagenet64_l2 checkpoint.
        >>> model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
        >>> pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        >>> pipe.to(device)

        >>> # Onestep Sampling
        >>> image = pipe(num_inference_steps=1).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample.png")

        >>> # Onestep sampling, class-conditional image generation
        >>> # ImageNet-64 class label 145 corresponds to king penguins
        >>> image = pipe(num_inference_steps=1, class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

        >>> # Multistep sampling, class-conditional image generation
        >>> # Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo:
        >>> # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
        >>> image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
        ```
"""


class ConsistencyModelPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional or class-conditional image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            compatible with [`CMStochasticIterativeScheduler`].
    """

    model_cpu_offload_seq = "unet"



    # Follows diffusers.VaeImageProcessor.postprocess



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)