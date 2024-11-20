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

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    logging,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.testing_utils import CaptureLogger


class SampleObject(ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SampleObject2(ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SampleObject3(ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SampleObject4(ConfigMixin):
    config_name = "config.json"

    @register_to_config


class ConfigTester(unittest.TestCase):








