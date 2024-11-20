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

import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet3DConditionModel,
    VideoToVideoSDPipeline,
)
from diffusers.utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    is_flaky,
    nightly,
    numpy_cosine_similarity_distance,
    skip_mps,
    torch_device,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


@skip_mps
class VideoToVideoSDPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = VideoToVideoSDPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS.union({"video"}) - {"image", "width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS.union({"video"}) - {"image"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    test_attention_slicing = False

    # No `output_type`.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )




    @is_flaky()

    @is_flaky()

    @is_flaky()

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")

    # (todo): sayakpaul
    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")



@nightly
@skip_mps
class VideoToVideoSDPipelineSlowTests(unittest.TestCase):