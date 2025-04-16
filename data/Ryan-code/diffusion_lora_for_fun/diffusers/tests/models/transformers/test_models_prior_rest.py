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
import inspect
import unittest

import torch
from parameterized import parameterized

from diffusers import PriorTransformer
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    slow,
    torch_all_close,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin


enable_full_determinism()


class PriorTransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = PriorTransformer
    main_input_name = "hidden_states"

    @property


    @property

    @property






@slow
class PriorTransformerIntegrationTests(unittest.TestCase):


    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.5861, 0.1283, -0.0931, 0.0882, 0.4476, 0.1329, -0.0498, 0.0640]],
            [37, [-0.4913, 0.0110, -0.0483, 0.0541, 0.4954, -0.0170, 0.0354, 0.1651]],
            # fmt: on
        ]
    )