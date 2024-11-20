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

from diffusers import AmusedPipeline, AmusedScheduler, UVit2DModel, VQModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AmusedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AmusedPipeline
    params = TEXT_TO_IMAGE_PARAMS | {"encoder_hidden_states", "negative_encoder_hidden_states"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS




    @unittest.skip("aMUSEd does not support lists of generators")


@slow
@require_torch_gpu
class AmusedPipelineSlowTests(unittest.TestCase):


