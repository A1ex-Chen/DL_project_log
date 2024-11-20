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
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import RepoCard
from safetensors.torch import load_file

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_accelerate_available
from diffusers.utils.testing_utils import (
    load_image,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_torch_gpu,
    slow,
    torch_device,
)


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


if is_accelerate_available():
    from accelerate.utils import release_memory


class StableDiffusionLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    scheduler_cls = DDIMScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
    }
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": 32,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
    }



    # Keeping this test here makes sense because it doesn't look any integration
    # (value assertions on logits).
    @slow
    @require_torch_gpu

    @require_torch_gpu


@slow
@require_torch_gpu
@require_peft_backend
class LoraIntegrationTests(unittest.TestCase):

















