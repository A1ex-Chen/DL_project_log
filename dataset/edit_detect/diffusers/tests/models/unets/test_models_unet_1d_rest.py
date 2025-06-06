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

import torch

from diffusers import UNet1DModel
from diffusers.utils.testing_utils import (
    backend_manual_seed,
    floats_tensor,
    slow,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


class UNet1DModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet1DModel
    main_input_name = "sample"

    @property

    @property

    @property













    @slow


class UNetRLModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet1DModel
    main_input_name = "sample"

    @property

    @property

    @property











