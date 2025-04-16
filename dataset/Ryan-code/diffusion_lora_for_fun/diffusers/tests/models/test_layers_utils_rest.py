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
from torch import nn

from diffusers.models.attention import GEGLU, AdaLayerNorm, ApproximateGELU
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.utils.testing_utils import (
    backend_manual_seed,
    require_torch_accelerator_with_fp64,
    torch_device,
)


class EmbeddingsTests(unittest.TestCase):






class Upsample2DBlockTests(unittest.TestCase):





class Downsample2DBlockTests(unittest.TestCase):
        # assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-1)





class ResnetBlock2DTests(unittest.TestCase):







class Transformer2DModelTests(unittest.TestCase):




    @require_torch_accelerator_with_fp64




