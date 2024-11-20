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

import contextlib
import gc
import inspect
import io
import re
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, TextToVideoZeroSDXLPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_accelerate_available, is_accelerate_version
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin, PipelineTesterMixin


enable_full_determinism()




class TextToVideoZeroSDXLPipelineFastTests(PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase):
    pipeline_class = TextToVideoZeroSDXLPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    generator_device = "cpu"





    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )



    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")

    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.17.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.17.0` or higher",
    )

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")


    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )


@nightly
@require_torch_gpu
class TextToVideoZeroSDXLPipelineSlowTests(unittest.TestCase):

