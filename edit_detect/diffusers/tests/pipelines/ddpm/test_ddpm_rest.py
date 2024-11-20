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

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device


enable_full_determinism()


class DDPMPipelineFastTests(unittest.TestCase):
    @property




@slow
@require_torch_gpu
class DDPMPipelineIntegrationTests(unittest.TestCase):