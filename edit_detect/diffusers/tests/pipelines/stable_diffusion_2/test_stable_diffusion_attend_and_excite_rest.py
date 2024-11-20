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

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionAttendAndExcitePipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    skip_mps,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineFromPipeTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


torch.backends.cuda.matmul.allow_tf32 = False


@skip_mps
class StableDiffusionAttendAndExcitePipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionAttendAndExcitePipeline
    test_attention_slicing = False
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS.union({"token_indices"})
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    # Attend and excite requires being able to run a backward pass at
    # inference time. There's no deterministic backward operator for pad

    @classmethod

    @classmethod













@require_torch_gpu
@nightly
class StableDiffusionAttendAndExcitePipelineIntegrationTests(unittest.TestCase):
    # Attend and excite requires being able to run a backward pass at
    # inference time. There's no deterministic backward operator for pad

    @classmethod

    @classmethod


