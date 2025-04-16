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
from transformers import (
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionGLIGENTextImagePipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import CLIPImageProjection
from diffusers.utils import load_image
from diffusers.utils.testing_utils import enable_full_determinism, torch_device

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineFromPipeTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class GligenTextImagePipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionGLIGENTextImagePipeline
    params = TEXT_TO_IMAGE_PARAMS | {"gligen_phrases", "gligen_images", "gligen_boxes"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS






