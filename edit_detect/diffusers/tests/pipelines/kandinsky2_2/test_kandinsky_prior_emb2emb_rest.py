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

import random
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import KandinskyV22PriorEmb2EmbPipeline, PriorTransformer, UnCLIPScheduler
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, skip_mps, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KandinskyV22PriorEmb2EmbPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22PriorEmb2EmbPipeline
    params = ["prompt", "image"]
    batch_params = ["prompt", "image"]
    required_optional_params = [
        "num_images_per_prompt",
        "strength",
        "generator",
        "num_inference_steps",
        "negative_prompt",
        "guidance_scale",
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

    @property

    @property




    @skip_mps

    @skip_mps