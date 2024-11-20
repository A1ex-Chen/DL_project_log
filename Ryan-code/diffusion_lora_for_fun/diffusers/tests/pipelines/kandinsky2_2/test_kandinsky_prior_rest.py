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

import inspect
import unittest

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import KandinskyV22PriorPipeline, PriorTransformer, UnCLIPScheduler
from diffusers.utils.testing_utils import enable_full_determinism, skip_mps, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Dummies:
    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property




class KandinskyV22PriorPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22PriorPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "generator",
        "num_inference_steps",
        "latents",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    callback_cfg_params = ["prompt_embeds", "text_encoder_hidden_states", "text_mask"]
    test_xformers_attention = False




    @skip_mps

    @skip_mps

    # override default test because no output_type "latent", use "pt" instead

        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["num_inference_steps"] = 2
        inputs["output_type"] = "pt"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0