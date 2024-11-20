# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import gc
import random
import unittest

import numpy as np
import torch

from diffusers import (
    DDIMScheduler,
    KandinskyV22ControlnetPipeline,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
    VQModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KandinskyV22ControlnetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22ControlnetPipeline
    params = ["image_embeds", "negative_image_embeds", "hint"]
    batch_params = ["image_embeds", "negative_image_embeds", "hint"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property







@nightly
@require_torch_gpu
class KandinskyV22ControlnetPipelineIntegrationTests(unittest.TestCase):

