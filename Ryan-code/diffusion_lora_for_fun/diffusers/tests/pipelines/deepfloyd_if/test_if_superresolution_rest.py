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

import torch

from diffusers import IFSuperResolutionPipeline
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import floats_tensor, load_numpy, require_torch_gpu, skip_mps, slow, torch_device

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from . import IFPipelineTesterMixin


@skip_mps
class IFSuperResolutionPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    pipeline_class = IFSuperResolutionPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}



    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )


    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")





@slow
@require_torch_gpu
class IFSuperResolutionPipelineSlowTests(unittest.TestCase):

