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
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    DPTConfig,
    DPTFeatureExtractor,
    DPTForDepthEstimation,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionDepth2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import is_accelerate_available, is_accelerate_version
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


@skip_mps
class StableDiffusionDepth2ImgPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionDepth2ImgPipeline
    test_save_load_optional_components = False
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"depth_mask"})




    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )







    @skip_mps



@slow
@require_torch_gpu
class StableDiffusionDepth2ImgPipelineSlowTests(unittest.TestCase):









@nightly
@require_torch_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):







        callback_fn.has_been_called = False

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.9 GB is allocated
        assert mem_bytes < 2.9 * 10**9


@nightly
@require_torch_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_depth2img_pndm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_depth2img_ddim(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3