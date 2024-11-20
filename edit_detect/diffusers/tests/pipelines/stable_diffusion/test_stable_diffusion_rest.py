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
import tempfile
import time
import traceback
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate_version_greater,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    require_torch_multi_gpu,
    run_test_in_subprocess,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# Will be run via run_test_in_subprocess


class StableDiffusionPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS























    # MPS currently doesn't support ComplexFloats, which are required for freeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps





@slow
@require_torch_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):



















    @require_python39_or_higher
    @require_torch_2



@slow
@require_torch_gpu
class StableDiffusionPipelineCkptTests(unittest.TestCase):





@nightly
@require_torch_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):









# (sayakpaul): This test suite was run in the DGX with two GPUs (1, 2).
@slow
@require_torch_multi_gpu
@require_accelerate_version_greater("0.27.0")
class StableDiffusionPipelineDeviceMapTests(unittest.TestCase):










        output_interrupted = sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)


@slow
@require_torch_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.3970, 0.3866, 0.4394, 0.4356, 0.4059])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_v1_4_with_freeu(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        image = sd_pipe(**inputs).images
        image = image[0, -3:, -3:, -1].flatten()
        expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 0.0344, 0.0309]
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5740, 0.4784, 0.3162, 0.6358, 0.5831, 0.5505, 0.5082, 0.5631, 0.5575])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10542, 0.09620, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config,
            final_sigmas_type="sigma_min",
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_attention_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.unet.set_default_attn_processor()
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        pipe.unet.set_default_attn_processor()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-3

    def test_stable_diffusion_vae_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        # enable vae slicing
        pipe.enable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 4 GB is allocated
        assert mem_bytes < 4e9

        # disable vae slicing
        pipe.disable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image = pipe(**inputs).images

        # make sure that more than 4 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 4e9
        # There is a small discrepancy at the image borders vs. a fully batched version.
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_vae_tiling(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        prompt = "a photograph of an astronaut riding a horse"

        # enable vae tiling
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output_chunked = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image_chunked = output_chunked.images

        mem_bytes = torch.cuda.max_memory_allocated()

        # disable vae tiling
        pipe.disable_vae_tiling()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images

        assert mem_bytes < 1e10
        max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_fp16_vs_autocast(self):
        # this test makes sure that the original model with autocast
        # and the new model with fp16 yield the same result
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_fp16 = pipe(**inputs).images

        with torch.autocast(torch_device):
            inputs = self.get_inputs(torch_device)
            image_autocast = pipe(**inputs).images

        # Make sure results are close enough
        diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 2e-2

    def test_stable_diffusion_intermediate_state(self):
        number_of_steps = 0



        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="np",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)


@slow
@require_torch_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.3970, 0.3866, 0.4394, 0.4356, 0.4059])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_v1_4_with_freeu(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 25

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        image = sd_pipe(**inputs).images
        image = image[0, -3:, -3:, -1].flatten()
        expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 0.0344, 0.0309]
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5740, 0.4784, 0.3162, 0.6358, 0.5831, 0.5505, 0.5082, 0.5631, 0.5575])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.10542, 0.09620, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config,
            final_sigmas_type="sigma_min",
        )
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_attention_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.unet.set_default_attn_processor()
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 3.75 GB is allocated
        assert mem_bytes < 3.75 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        pipe.unet.set_default_attn_processor()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.75 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 3.75 * 10**9
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-3

    def test_stable_diffusion_vae_slicing(self):
        torch.cuda.reset_peak_memory_stats()
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        # enable vae slicing
        pipe.enable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image_sliced = pipe(**inputs).images

        mem_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # make sure that less than 4 GB is allocated
        assert mem_bytes < 4e9

        # disable vae slicing
        pipe.disable_vae_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        inputs["prompt"] = [inputs["prompt"]] * 4
        inputs["latents"] = torch.cat([inputs["latents"]] * 4)
        image = pipe(**inputs).images

        # make sure that more than 4 GB is allocated
        mem_bytes = torch.cuda.max_memory_allocated()
        assert mem_bytes > 4e9
        # There is a small discrepancy at the image borders vs. a fully batched version.
        max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_vae_tiling(self):
        torch.cuda.reset_peak_memory_stats()
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        prompt = "a photograph of an astronaut riding a horse"

        # enable vae tiling
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output_chunked = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image_chunked = output_chunked.images

        mem_bytes = torch.cuda.max_memory_allocated()

        # disable vae tiling
        pipe.disable_vae_tiling()
        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            [prompt],
            width=1024,
            height=1024,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images

        assert mem_bytes < 1e10
        max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(), image.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_fp16_vs_autocast(self):
        # this test makes sure that the original model with autocast
        # and the new model with fp16 yield the same result
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_fp16 = pipe(**inputs).images

        with torch.autocast(torch_device):
            inputs = self.get_inputs(torch_device)
            image_autocast = pipe(**inputs).images

        # Make sure results are close enough
        diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
        # They ARE different since ops are not run always at the same precision
        # however, they should be extremely close.
        assert diff.mean() < 2e-2

    def test_stable_diffusion_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.5693, -0.3018, -0.9746, 0.0518, -0.8770, 0.7559, -1.7402, 0.1022, 1.1582]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.1958, -0.2993, -1.0166, -0.5005, -0.4810, 0.6162, -0.9492, 0.6621, 1.4492]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    def test_stable_diffusion_low_cpu_mem_usage(self):
        pipeline_id = "CompVis/stable-diffusion-v1-4"

        start_time = time.time()
        pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16)
        pipeline_low_cpu_mem_usage.to(torch_device)
        low_cpu_mem_usage_time = time.time() - start_time

        start_time = time.time()
        _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=torch.float16, low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time

        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.8 GB is allocated
        assert mem_bytes < 2.8 * 10**9

    def test_stable_diffusion_pipeline_with_model_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        # Normal inference

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        outputs = pipe(**inputs)
        mem_bytes = torch.cuda.max_memory_allocated()

        # With model offloading

        # Reload but don't move to cuda
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
        pipe.unet.set_default_attn_processor()

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        outputs_offloaded = pipe(**inputs)
        mem_bytes_offloaded = torch.cuda.max_memory_allocated()

        images = outputs.images
        offloaded_images = outputs_offloaded.images

        max_diff = numpy_cosine_similarity_distance(images.flatten(), offloaded_images.flatten())
        assert max_diff < 1e-3
        assert mem_bytes_offloaded < mem_bytes
        assert mem_bytes_offloaded < 3.5 * 10**9
        for module in pipe.text_encoder, pipe.unet, pipe.vae:
            assert module.device == torch.device("cpu")

        # With attention slicing
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe.enable_attention_slicing()
        _ = pipe(**inputs)
        mem_bytes_slicing = torch.cuda.max_memory_allocated()

        assert mem_bytes_slicing < mem_bytes_offloaded
        assert mem_bytes_slicing < 3 * 10**9

    def test_stable_diffusion_textual_inversion(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)
        pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_model_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_model_cpu_offload()
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    def test_stable_diffusion_textual_inversion_with_sequential_cpu_offload(self):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.enable_sequential_cpu_offload()
        pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

        a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
        a111_file_neg = hf_hub_download(
            "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
        )
        pipe.load_textual_inversion(a111_file)
        pipe.load_textual_inversion(a111_file_neg)

        generator = torch.Generator(device="cpu").manual_seed(1)

        prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
        neg_prompt = "Style-Winter-neg"

        image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
        )

        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 8e-1

    @require_python39_or_higher
    @require_torch_2
    def test_stable_diffusion_compile(self):
        seed = 0
        inputs = self.get_inputs(torch_device, seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["torch_device"] = torch_device
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_stable_diffusion_compile, inputs=inputs)

    def test_stable_diffusion_lcm(self):
        unet = UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet")
        sd_pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", unet=unet).to(torch_device)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 6
        inputs["output_type"] = "pil"

        image = sd_pipe(**inputs).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_lcm.png"
        )

        image = sd_pipe.image_processor.pil_to_numpy(image)
        expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())

        assert max_diff < 1e-2


@slow
@require_torch_gpu
class StableDiffusionPipelineCkptTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_download_from_hub(self):
        ckpt_paths = [
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors",
        ]

        for ckpt_path in ckpt_paths:
            pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)

    def test_download_local(self):
        ckpt_filename = hf_hub_download("runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.safetensors")
        config_filename = hf_hub_download("runwayml/stable-diffusion-v1-5", filename="v1-inference.yaml")

        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_filename, config_files={"v1": config_filename}, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

        assert image_out.shape == (512, 512, 3)


@nightly
@require_torch_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 3e-3

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3


# (sayakpaul): This test suite was run in the DGX with two GPUs (1, 2).
@slow
@require_torch_multi_gpu
@require_accelerate_version_greater("0.27.0")
class StableDiffusionPipelineDeviceMapTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, generator_device="cpu", seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def get_pipeline_output_without_device_map(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(torch_device)
        sd_pipe.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        no_device_map_image = sd_pipe(**inputs).images

        del sd_pipe

        return no_device_map_image

    def test_forward_pass_balanced_device_map(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_components_put_in_right_devices(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )

        assert len(set(sd_pipe_with_device_map.hf_device_map.values())) >= 2

    def test_max_memory(self):
        no_device_map_image = self.get_pipeline_output_without_device_map()

        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            device_map="balanced",
            max_memory={0: "1GB", 1: "1GB"},
            torch_dtype=torch.float16,
        )
        sd_pipe_with_device_map.set_progress_bar_config(disable=True)
        inputs = self.get_inputs()
        device_map_image = sd_pipe_with_device_map(**inputs).images

        max_diff = np.abs(device_map_image - no_device_map_image).max()
        assert max_diff < 1e-3

    def test_reset_device_map(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        for name, component in sd_pipe_with_device_map.components.items():
            if isinstance(component, torch.nn.Module):
                assert component.device.type == "cpu"

    def test_reset_device_map_to(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `to()` can be used and the pipeline can be called.
        pipe = sd_pipe_with_device_map.to("cuda")
        _ = pipe("hello", num_inference_steps=2)

    def test_reset_device_map_enable_model_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_model_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_model_cpu_offload()
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)

    def test_reset_device_map_enable_sequential_cpu_offload(self):
        sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", device_map="balanced", torch_dtype=torch.float16
        )
        sd_pipe_with_device_map.reset_device_map()

        assert sd_pipe_with_device_map.hf_device_map is None

        # Make sure `enable_sequential_cpu_offload()` can be used and the pipeline can be called.
        sd_pipe_with_device_map.enable_sequential_cpu_offload()
        _ = sd_pipe_with_device_map("hello", num_inference_steps=2)