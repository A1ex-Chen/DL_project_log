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
import traceback
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    skip_mps,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# Will be run via run_test_in_subprocess


class StableDiffusionImg2ImgPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS












    @skip_mps

    @skip_mps

    @skip_mps

    @skip_mps





@slow
@require_torch_gpu
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):












    @require_python39_or_higher
    @require_torch_2


@nightly
@require_torch_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):







        output_interrupted = sd_pipe(
            prompt,
            image=inputs["image"],
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
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):
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
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_img2img_default(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4300, 0.4662, 0.4930, 0.3990, 0.4307, 0.4525, 0.3719, 0.4064, 0.3923])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0389, 0.0346, 0.0415, 0.0290, 0.0218, 0.0210, 0.0408, 0.0567, 0.0271])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_ddim(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0593, 0.0607, 0.0851, 0.0582, 0.0636, 0.0721, 0.0751, 0.0981, 0.0781])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0



        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            image=inputs["image"],
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
            image=inputs["image"],
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
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):
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
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_img2img_default(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.4300, 0.4662, 0.4930, 0.3990, 0.4307, 0.4525, 0.3719, 0.4064, 0.3923])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_k_lms(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0389, 0.0346, 0.0415, 0.0290, 0.0218, 0.0210, 0.0408, 0.0567, 0.0271])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_ddim(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0593, 0.0607, 0.0851, 0.0582, 0.0636, 0.0721, 0.0751, 0.0981, 0.0781])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.4958, 0.5107, 1.1045, 2.7539, 4.6680, 3.8320, 1.5049, 1.8633, 2.6523])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.4956, 0.5078, 1.0918, 2.7520, 4.6484, 3.8125, 1.5146, 1.8633, 2.6367])

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9

    def test_stable_diffusion_pipeline_with_model_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)

        # Normal inference

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe(**inputs)
        mem_bytes = torch.cuda.max_memory_allocated()

        # With model offloading

        # Reload but don't move to cuda
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            safety_checker=None,
            torch_dtype=torch.float16,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        _ = pipe(**inputs)
        mem_bytes_offloaded = torch.cuda.max_memory_allocated()

        assert mem_bytes_offloaded < mem_bytes
        for module in pipe.text_encoder, pipe.unet, pipe.vae:
            assert module.device == torch.device("cpu")

    def test_img2img_2nd_order(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = HeunDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 10
        inputs["strength"] = 0.75
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/img2img_heun.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 5e-2

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 11
        inputs["strength"] = 0.75
        image_other = sd_pipe(**inputs).images[0]

        mean_diff = np.abs(image - image_other).mean()

        # images should be very similar
        assert mean_diff < 5e-2

    def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        )
        # resize to resolution that is divisible by 8 but not 16 or 32
        init_image = init_image.resize((760, 504))

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "A fantasy landscape, trending on artstation"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.75,
            guidance_scale=7.5,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        image_slice = image[255:258, 383:386, -1]

        assert image.shape == (504, 760, 3)
        expected_slice = np.array([0.9393, 0.9500, 0.9399, 0.9438, 0.9458, 0.9400, 0.9455, 0.9414, 0.9423])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3

    def test_img2img_safety_checker_works(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 20
        # make sure the safety checker is activated
        inputs["prompt"] = "naked, sex, porn"
        out = sd_pipe(**inputs)

        assert out.nsfw_content_detected[0], f"Safety checker should work for prompt: {inputs['prompt']}"
        assert np.abs(out.images[0]).sum() < 1e-5  # should be all zeros

    @require_python39_or_higher
    @require_torch_2
    def test_img2img_compile(self):
        seed = 0
        inputs = self.get_inputs(torch_device, seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["torch_device"] = torch_device
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_img2img_compile, inputs=inputs)


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

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 50,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_img2img_pndm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_ddim(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/stable_diffusion_1_5_dpm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3