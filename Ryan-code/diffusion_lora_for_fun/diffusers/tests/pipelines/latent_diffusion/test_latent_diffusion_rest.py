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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline, UNet2DConditionModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_numpy,
    nightly,
    require_torch_gpu,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class LDMTextToImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LDMTextToImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {
        "negative_prompt",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "num_images_per_prompt",
        "callback",
        "callback_steps",
    }
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS





@nightly
@require_torch_gpu
class LDMTextToImagePipelineSlowTests(unittest.TestCase):





@nightly
@require_torch_gpu
class LDMTextToImagePipelineNightlyTests(unittest.TestCase):


