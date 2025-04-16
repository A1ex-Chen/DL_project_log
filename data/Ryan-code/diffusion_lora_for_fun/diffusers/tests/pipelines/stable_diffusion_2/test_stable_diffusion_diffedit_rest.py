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
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionDiffEditPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionDiffEditPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionDiffEditPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS - {"height", "width", "image"} | {"image_latents"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS - {"image"} | {"image_latents"}
    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])











@require_torch_gpu
@nightly
class StableDiffusionDiffEditPipelineIntegrationTests(unittest.TestCase):


    @classmethod



@nightly
@require_torch_gpu
class StableDiffusionDiffEditPipelineNightlyTests(unittest.TestCase):


    @classmethod
