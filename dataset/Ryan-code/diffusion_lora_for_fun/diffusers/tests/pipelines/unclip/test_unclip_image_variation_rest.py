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
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DiffusionPipeline,
    UnCLIPImageVariationPipeline,
    UnCLIPScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_gpu,
    skip_mps,
    torch_device,
)

from ..pipeline_params import IMAGE_VARIATION_BATCH_PARAMS, IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference


enable_full_determinism()


class UnCLIPImageVariationPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UnCLIPImageVariationPipeline
    params = IMAGE_VARIATION_PARAMS - {"height", "width", "guidance_scale"}
    batch_params = IMAGE_VARIATION_BATCH_PARAMS

    required_optional_params = [
        "generator",
        "return_dict",
        "decoder_num_inference_steps",
        "super_res_num_inference_steps",
    ]
    test_xformers_attention = False

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property







    # Overriding PipelineTesterMixin::test_attention_slicing_forward_pass
    # because UnCLIP GPU undeterminism requires a looser check.
    @skip_mps

    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because UnCLIP undeterminism requires a looser check.
    @unittest.skip("UnCLIP produces very large differences. Test is not useful.")
    @skip_mps


    @skip_mps

    @unittest.skip("UnCLIP produces very large difference. Test is not useful.")
    @skip_mps

    @skip_mps

    @unittest.skip("UnCLIP produces very large difference in fp16 vs fp32. Test is not useful.")


@nightly
@require_torch_gpu
class UnCLIPImageVariationPipelineIntegrationTests(unittest.TestCase):

