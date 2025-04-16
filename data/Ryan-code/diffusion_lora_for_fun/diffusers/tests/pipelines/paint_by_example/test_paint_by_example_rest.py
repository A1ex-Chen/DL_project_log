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

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig

from diffusers import AutoencoderKL, PaintByExamplePipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    nightly,
    require_torch_gpu,
    torch_device,
)

from ..pipeline_params import IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class PaintByExamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = PaintByExamplePipeline
    params = IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])  # TO_DO: update the image_prams once refactored VaeImageProcessor.preprocess








@nightly
@require_torch_gpu
class PaintByExamplePipelineIntegrationTests(unittest.TestCase):

