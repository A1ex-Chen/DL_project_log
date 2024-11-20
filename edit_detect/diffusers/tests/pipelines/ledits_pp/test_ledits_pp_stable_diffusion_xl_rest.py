# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LEditsPPPipelineStableDiffusionXL,
    UNet2DConditionModel,
)

# from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)


enable_full_determinism()


@skip_mps
class LEditsPPPipelineStableDiffusionXLFastTests(unittest.TestCase):
    pipeline_class = LEditsPPPipelineStableDiffusionXL








@slow
@require_torch_gpu
class LEditsPPPipelineStableDiffusionXLSlowTests(unittest.TestCase):
    @classmethod
