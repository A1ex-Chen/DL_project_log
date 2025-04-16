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
import random
import unittest

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    I2VGenXLPipeline,
)
from diffusers.models.unets import I2VGenXLUNet
from diffusers.utils import is_xformers_available, load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    numpy_cosine_similarity_distance,
    print_tensor_test,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin, SDFunctionTesterMixin


enable_full_determinism()


@skip_mps
class I2VGenXLPipelineFastTests(SDFunctionTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = I2VGenXLPipeline
    params = frozenset(["prompt", "negative_prompt", "image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "generator"])
    # No `output_type`.
    required_optional_params = frozenset(["num_inference_steps", "generator", "latents", "return_dict"])








    @unittest.skip("Deprecated functionality")

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )





@slow
@require_torch_gpu
class I2VGenXLPipelineSlowTests(unittest.TestCase):

