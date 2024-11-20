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
import unittest

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DPMSolverMultistepScheduler, Transformer2DModel
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import enable_full_determinism, load_numpy, nightly, require_torch_gpu, torch_device

from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class DiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DiTPipeline
    params = CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS





    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )


@nightly
@require_torch_gpu
class DiTPipelineIntegrationTests(unittest.TestCase):


