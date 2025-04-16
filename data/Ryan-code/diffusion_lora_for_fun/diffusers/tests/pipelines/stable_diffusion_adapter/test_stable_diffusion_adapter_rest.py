# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import numpy as np
import torch
from parameterized import parameterized
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    MultiAdapter,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin, PipelineTesterMixin, assert_mean_pixel_difference


enable_full_determinism()


class AdapterTests:
    pipeline_class = StableDiffusionAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS





    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )


    @parameterized.expand(
        [
            # (dim=264) The internal feature map will be 33x33 after initial pixel unshuffling (downscaled x8).
            (((4 * 8 + 1) * 8),),
            # (dim=272) The internal feature map will be 17x17 after the first T2I down block (downscaled x16).
            (((4 * 4 + 1) * 16),),
            # (dim=288) The internal feature map will be 9x9 after the second T2I down block (downscaled x32).
            (((4 * 2 + 1) * 32),),
            # (dim=320) The internal feature map will be 5x5 after the third T2I down block (downscaled x64).
            (((4 * 1 + 1) * 64),),
        ]
    )




class StableDiffusionFullAdapterPipelineFastTests(
    AdapterTests, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase
):




class StableDiffusionLightAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):




class StableDiffusionMultiAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):








@slow
@require_torch_gpu
class StableDiffusionAdapterPipelineSlowTests(unittest.TestCase):













