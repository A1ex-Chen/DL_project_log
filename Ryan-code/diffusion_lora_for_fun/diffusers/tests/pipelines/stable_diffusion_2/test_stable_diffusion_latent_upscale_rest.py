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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()




class StableDiffusionLatentUpscalePipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
        "negative_prompt_embeds",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    @property














@require_torch_gpu
@slow
class StableDiffusionLatentUpscalePipelineIntegrationTests(unittest.TestCase):

