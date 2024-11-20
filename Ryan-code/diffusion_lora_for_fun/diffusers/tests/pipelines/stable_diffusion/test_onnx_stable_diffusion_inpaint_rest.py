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

from diffusers import LMSDiscreteScheduler, OnnxStableDiffusionInpaintPipeline
from diffusers.utils.testing_utils import (
    is_onnx_available,
    load_image,
    nightly,
    require_onnxruntime,
    require_torch_gpu,
)

from ..test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


@nightly
@require_onnxruntime
@require_torch_gpu
class OnnxStableDiffusionInpaintPipelineIntegrationTests(unittest.TestCase):
    @property

    @property

