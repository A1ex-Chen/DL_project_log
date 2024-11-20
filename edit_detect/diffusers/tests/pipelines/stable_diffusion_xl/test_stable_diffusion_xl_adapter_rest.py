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
from parameterized import parameterized
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    LCMScheduler,
    MultiAdapter,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
    assert_mean_pixel_difference,
)


enable_full_determinism()


class StableDiffusionXLAdapterPipelineFastTests(
    IPAdapterTesterMixin, PipelineTesterMixin, SDXLOptionalComponentsTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionXLAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS






    @parameterized.expand(
        [
            # (dim=144) The internal feature map will be 9x9 after initial pixel unshuffling (downscaled x16).
            (((4 * 2 + 1) * 16),),
            # (dim=160) The internal feature map will be 5x5 after the first T2I down block (downscaled x32).
            (((4 * 1 + 1) * 32),),
        ]
    )

    @parameterized.expand(["full_adapter", "full_adapter_xl", "light_adapter"])





class StableDiffusionXLMultiAdapterPipelineFastTests(
    StableDiffusionXLAdapterPipelineFastTests, PipelineTesterMixin, unittest.TestCase
):








