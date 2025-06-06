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
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import DDPMWuerstchenScheduler, StableCascadePriorPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models import StableCascadeUNet
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()




class StableCascadePriorPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableCascadePriorPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "generator",
        "num_inference_steps",
        "latents",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False
    callback_cfg_params = ["text_encoder_hidden_states"]

    @property

    @property

    @property

    @property

    @property

    @property

    @property




    @skip_mps

    @skip_mps

    @unittest.skip(reason="fp16 not supported")



    @require_peft_backend
    @unittest.skip(reason="no lora support for now")



@slow
@require_torch_gpu
class StableCascadePriorPipelineIntegrationTests(unittest.TestCase):

