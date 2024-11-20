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
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionInstructPix2PixPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionInstructPix2PixPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width", "cross_attention_kwargs"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"image_latents"}) - {"negative_prompt_embeds"}








    # Overwrite the default test_latents_inputs because pix2pix encode the image differently

    # Override the default test_callback_cfg because pix2pix create inputs for cfg differently


@slow
@require_torch_gpu
class StableDiffusionInstructPix2PixPipelineSlowTests(unittest.TestCase):









        inputs = self.get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.0
        inputs["num_inference_steps"] = 2
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        inputs["callback_on_step_end"] = callback_no_cfg
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        out_callback_no_cfg = pipe(**inputs)[0]

        assert out_no_cfg.shape == out_callback_no_cfg.shape


@slow
@require_torch_gpu
class StableDiffusionInstructPix2PixPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)
        image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_pix2pix/example.jpg"
        )
        inputs = {
            "prompt": "turn him into a cyborg",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_pix2pix_default(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5902, 0.6015, 0.6027, 0.5983, 0.6092, 0.6061, 0.5765, 0.5785, 0.5555])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_k_lms(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.6578, 0.6817, 0.6972, 0.6761, 0.6856, 0.6916, 0.6428, 0.6516, 0.6301])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_ddim(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3828, 0.3834, 0.3818, 0.3792, 0.3865, 0.3752, 0.3792, 0.3847, 0.3753])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_pix2pix_intermediate_state(self):
        number_of_steps = 0


        callback_fn.has_been_called = False

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9

    def test_stable_diffusion_pix2pix_pipeline_multiple_of_8(self):
        inputs = self.get_inputs()
        # resize to resolution that is divisible by 8 but not 16 or 32
        inputs["image"] = inputs["image"].resize((504, 504))

        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        output = pipe(**inputs)
        image = output.images[0]

        image_slice = image[255:258, 383:386, -1]

        assert image.shape == (504, 504, 3)
        expected_slice = np.array([0.2726, 0.2529, 0.2664, 0.2655, 0.2641, 0.2642, 0.2591, 0.2649, 0.2590])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3