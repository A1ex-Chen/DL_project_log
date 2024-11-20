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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import DDPMWuerstchenScheduler, WuerstchenCombinedPipeline
from diffusers.pipelines.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt, WuerstchenPrior
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class WuerstchenCombinedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WuerstchenCombinedPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "prior_guidance_scale",
        "decoder_guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "prior_num_inference_steps",
        "output_type",
    ]
    test_xformers_attention = True

    @property

    @property

    @property

    @property

    @property

    @property

    @property




    @require_torch_gpu


    @unittest.skip(reason="flakey and float16 requires CUDA")

