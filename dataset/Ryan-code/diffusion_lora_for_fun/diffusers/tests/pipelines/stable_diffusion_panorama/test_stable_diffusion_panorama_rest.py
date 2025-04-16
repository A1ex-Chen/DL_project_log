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
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPanoramaPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, skip_mps, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


@skip_mps
class StableDiffusionPanoramaPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionPanoramaPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS





    # override to speed the overall test timing up.

    # override to speed the overall test timing up.








@nightly
@require_torch_gpu
class StableDiffusionPanoramaNightlyTests(unittest.TestCase):







        callback_fn.has_been_called = False

        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3

    def test_stable_diffusion_panorama_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 5.2 GB is allocated
        assert mem_bytes < 5.5 * 10**9