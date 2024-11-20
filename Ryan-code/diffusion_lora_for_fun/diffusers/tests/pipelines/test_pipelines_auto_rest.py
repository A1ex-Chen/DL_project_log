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
import os
import shutil
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    DiffusionPipeline,
)
from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
)
from diffusers.utils.testing_utils import slow


PRETRAINED_MODEL_REPO_MAPPING = OrderedDict(
    [
        ("stable-diffusion", "runwayml/stable-diffusion-v1-5"),
        ("if", "DeepFloyd/IF-I-XL-v1.0"),
        ("kandinsky", "kandinsky-community/kandinsky-2-1"),
        ("kandinsky22", "kandinsky-community/kandinsky-2-2-decoder"),
    ]
)


class AutoPipelineFastTest(unittest.TestCase):
    @property











@slow
class AutoPipelineIntegrationTest(unittest.TestCase):

