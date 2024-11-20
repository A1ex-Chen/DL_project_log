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

import torch

from diffusers import (
    AutoencoderKL,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_hf_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)


enable_full_determinism()


@slow
@require_torch_gpu
class AutoencoderKLSingleFileTests(unittest.TestCase):
    model_class = AutoencoderKL
    ckpt_path = (
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    )
    repo_id = "stabilityai/sd-vae-ft-mse"
    main_input_name = "sample"
    base_precision = 1e-2






