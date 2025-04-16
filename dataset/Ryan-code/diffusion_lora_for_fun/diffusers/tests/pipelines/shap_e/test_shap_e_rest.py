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
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import HeunDiscreteScheduler, PriorTransformer, ShapEPipeline
from diffusers.pipelines.shap_e import ShapERenderer
from diffusers.utils.testing_utils import load_numpy, nightly, require_torch_gpu, torch_device

from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference


class ShapEPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ShapEPipeline
    params = ["prompt"]
    batch_params = ["prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "num_inference_steps",
        "generator",
        "latents",
        "guidance_scale",
        "frame_size",
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









    @unittest.skip("Key error is raised with accelerate")


@nightly
@require_torch_gpu
class ShapEPipelineIntegrationTests(unittest.TestCase):

