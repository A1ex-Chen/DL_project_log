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

import unittest

import numpy as np
import torch

from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class DDIMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DDIMPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "latents",
        "callback",
        "callback_steps",
    }
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS









@slow
@require_torch_gpu
class DDIMPipelineIntegrationTests(unittest.TestCase):
