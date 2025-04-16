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

import tempfile
import unittest

import torch

from diffusers import UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.utils.testing_utils import enable_full_determinism, skip_mps, torch_device


enable_full_determinism()


class EMAModelTests(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    batch_size = 1
    prompt_length = 77
    text_encoder_hidden_dim = 32
    num_in_channels = 4
    latent_height = latent_width = 64
    generator = torch.manual_seed(0)









    @skip_mps