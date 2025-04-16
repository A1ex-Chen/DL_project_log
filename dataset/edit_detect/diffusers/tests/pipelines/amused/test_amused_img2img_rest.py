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
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AmusedImg2ImgPipeline, AmusedScheduler, UVit2DModel, VQModel
from diffusers.utils import load_image
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AmusedImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AmusedImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width", "latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "latents",
    }




    @unittest.skip("aMUSEd does not support lists of generators")


@slow
@require_torch_gpu
class AmusedImg2ImgPipelineSlowTests(unittest.TestCase):


