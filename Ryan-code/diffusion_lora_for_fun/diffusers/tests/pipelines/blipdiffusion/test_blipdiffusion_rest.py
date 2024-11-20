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
from PIL import Image
from transformers import CLIPTokenizer
from transformers.models.blip_2.configuration_blip_2 import Blip2Config
from transformers.models.clip.configuration_clip import CLIPTextConfig

from diffusers import AutoencoderKL, BlipDiffusionPipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.utils.testing_utils import enable_full_determinism
from src.diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from src.diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class BlipDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BlipDiffusionPipeline
    params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    batch_params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "neg_prompt",
        "guidance_scale",
        "prompt_strength",
        "prompt_reps",
    ]


