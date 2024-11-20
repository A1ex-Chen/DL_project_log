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
import time
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)


enable_full_determinism()


class StableDiffusion2VPredictionPipelineFastTests(unittest.TestCase):


    @property

    @property

    @property



    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")


@slow
@require_torch_gpu
class StableDiffusion2VPredictionPipelineIntegrationTests(unittest.TestCase):














        test_callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Andromeda galaxy in a bottle"

        generator = torch.manual_seed(0)
        pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 20

    def test_stable_diffusion_low_cpu_mem_usage_v_pred(self):
        pipeline_id = "stabilityai/stable-diffusion-2"

        start_time = time.time()
        pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline_low_cpu_mem_usage.to(torch_device)
        low_cpu_mem_usage_time = time.time() - start_time

        start_time = time.time()
        _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time

        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading_v_pred(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipeline_id = "stabilityai/stable-diffusion-2"
        prompt = "Andromeda galaxy in a bottle"

        pipeline = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline = pipeline.to(torch_device)
        pipeline.enable_attention_slicing(1)
        pipeline.enable_sequential_cpu_offload()

        generator = torch.manual_seed(0)
        _ = pipeline(prompt, generator=generator, num_inference_steps=5)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.8 GB is allocated
        assert mem_bytes < 2.8 * 10**9