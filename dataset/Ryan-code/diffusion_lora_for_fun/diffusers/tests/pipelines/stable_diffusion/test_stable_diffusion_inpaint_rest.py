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
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
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
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
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


class StableDiffusionInpaintPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"mask", "masked_image_latents"})













class StableDiffusionSimpleInpaintPipelineFastTests(StableDiffusionInpaintPipelineFastTests):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess










@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineSlowTests(unittest.TestCase):








    @require_python39_or_higher
    @require_torch_2





@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineAsymmetricAutoencoderKLSlowTests(unittest.TestCase):








    @require_python39_or_higher
    @require_torch_2






@nightly
@require_torch_gpu
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):








class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):














        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            image=inputs["image"],
            mask_image=inputs["mask_image"],
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
            mask_image=inputs["mask_image"],
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

    def test_ip_adapter_single(self, from_simple=False, expected_pipe_slice=None):
        if not from_simple:
            expected_pipe_slice = None
            if torch_device == "cpu":
                expected_pipe_slice = np.array(
                    [0.4390, 0.5452, 0.3772, 0.5448, 0.6031, 0.4480, 0.5194, 0.4687, 0.4640]
                )
        return super().test_ip_adapter_single(expected_pipe_slice=expected_pipe_slice)


class StableDiffusionSimpleInpaintPipelineFastTests(StableDiffusionInpaintPipelineFastTests):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs_2images(self, device, seed=0, img_res=64):
        # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
        image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(device)
        image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed + 22)).to(device)
        # Convert images to [-1, 1]
        init_image1 = 2.0 * image1 - 1.0
        init_image2 = 2.0 * image2 - 1.0

        # empty mask
        mask_image = torch.zeros((1, 1, img_res, img_res), device=device)

        if str(device).startswith("mps"):
            generator1 = torch.manual_seed(seed)
            generator2 = torch.manual_seed(seed)
        else:
            generator1 = torch.Generator(device=device).manual_seed(seed)
            generator2 = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": ["A painting of a squirrel eating a burger"] * 2,
            "image": [init_image1, init_image2],
            "mask_image": [mask_image] * 2,
            "generator": [generator1, generator2],
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_ip_adapter_single(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.6345, 0.5395, 0.5611, 0.5403, 0.5830, 0.5855, 0.5193, 0.5443, 0.5211])
        return super().test_ip_adapter_single(from_simple=True, expected_pipe_slice=expected_pipe_slice)

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.6584, 0.5424, 0.5649, 0.5449, 0.5897, 0.6111, 0.5404, 0.5463, 0.5214])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.6240, 0.5355, 0.5649, 0.5378, 0.5374, 0.6242, 0.5132, 0.5347, 0.5396])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.6240, 0.5355, 0.5649, 0.5378, 0.5374, 0.6242, 0.5132, 0.5347, 0.5396])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_2_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # test to confirm if we pass two same image, we will get same output
        inputs = self.get_dummy_inputs(device)
        gen1 = torch.Generator(device=device).manual_seed(0)
        gen2 = torch.Generator(device=device).manual_seed(0)
        for name in ["prompt", "image", "mask_image"]:
            inputs[name] = [inputs[name]] * 2
        inputs["generator"] = [gen1, gen2]
        images = sd_pipe(**inputs).images

        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() < 1e-4

        # test to confirm that if we pass two different images, we will get different output
        inputs = self.get_dummy_inputs_2images(device)
        images = sd_pipe(**inputs).images
        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() > 1e-2

    def test_stable_diffusion_inpaint_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device, output_pil=False)
        half_dim = inputs["image"].shape[2] // 2
        inputs["mask_image"][0, 0, :half_dim, :half_dim] = 0

        inputs["num_inference_steps"] = 4
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array(
            [[0.6387283, 0.5564158, 0.58631873, 0.5539942, 0.5494673, 0.6461868, 0.5251618, 0.5497595, 0.5508756]]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-4


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0427, 0.0460, 0.0483, 0.0460, 0.0584, 0.0521, 0.1549, 0.1695, 0.1794])

        assert np.abs(expected_slice - image_slice).max() < 6e-4

    def test_stable_diffusion_inpaint_fp16(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1509, 0.1245, 0.1672, 0.1655, 0.1519, 0.1226, 0.1462, 0.1567, 0.2451])
        assert np.abs(expected_slice - image_slice).max() < 1e-1

    def test_stable_diffusion_inpaint_pndm(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0425, 0.0273, 0.0344, 0.1694, 0.1727, 0.1812, 0.3256, 0.3311, 0.3272])

        assert np.abs(expected_slice - image_slice).max() < 5e-3

    def test_stable_diffusion_inpaint_k_lms(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9314, 0.7575, 0.9432, 0.8885, 0.9028, 0.7298, 0.9811, 0.9667, 0.7633])

        assert np.abs(expected_slice - image_slice).max() < 6e-3

    def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype=torch.float16
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

    @require_python39_or_higher
    @require_torch_2
    def test_inpaint_compile(self):
        seed = 0
        inputs = self.get_inputs(torch_device, seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["torch_device"] = torch_device
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_inpaint_compile, inputs=inputs)

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input image to a random size (one that would cause a tensor mismatch error)
        inputs["image"] = inputs["image"].resize((127, 127))
        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        inputs["height"] = 128
        inputs["width"] = 128
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input strength
        inputs["strength"] = 0.75
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, 253:256, 253:256, -1].flatten()
        expected_slice = np.array([0.2728, 0.2803, 0.2665, 0.2511, 0.2774, 0.2586, 0.2391, 0.2392, 0.2582])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_simple_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images

        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3757, 0.3875, 0.4445, 0.4353, 0.3780, 0.4513, 0.3965, 0.3984, 0.4362])
        assert np.abs(expected_slice - image_slice).max() < 1e-3


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineAsymmetricAutoencoderKLSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.vae = vae
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0522, 0.0604, 0.0596, 0.0449, 0.0493, 0.0427, 0.1186, 0.1289, 0.1442])

        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_inpaint_fp16(self):
        vae = AsymmetricAutoencoderKL.from_pretrained(
            "cross-attention/asymmetric-autoencoder-kl-x-1-5", torch_dtype=torch.float16
        )
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae = vae
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1343, 0.1406, 0.1440, 0.1504, 0.1729, 0.0989, 0.1807, 0.2822, 0.1179])

        assert np.abs(expected_slice - image_slice).max() < 5e-2

    def test_stable_diffusion_inpaint_pndm(self):
        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae = vae
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0966, 0.1083, 0.1148, 0.1422, 0.1318, 0.1197, 0.3702, 0.3537, 0.3288])

        assert np.abs(expected_slice - image_slice).max() < 5e-3

    def test_stable_diffusion_inpaint_k_lms(self):
        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae = vae
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.8931, 0.8683, 0.8965, 0.8501, 0.8592, 0.9118, 0.8734, 0.7463, 0.8990])
        assert np.abs(expected_slice - image_slice).max() < 6e-3

    def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        vae = AsymmetricAutoencoderKL.from_pretrained(
            "cross-attention/asymmetric-autoencoder-kl-x-1-5", torch_dtype=torch.float16
        )
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.vae = vae
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.45 GB is allocated
        assert mem_bytes < 2.45 * 10**9

    @require_python39_or_higher
    @require_torch_2
    def test_inpaint_compile(self):
        pass

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        vae = AsymmetricAutoencoderKL.from_pretrained(
            "cross-attention/asymmetric-autoencoder-kl-x-1-5",
        )
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.vae = vae
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input image to a random size (one that would cause a tensor mismatch error)
        inputs["image"] = inputs["image"].resize((127, 127))
        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        inputs["height"] = 128
        inputs["width"] = 128
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.vae = vae
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input strength
        inputs["strength"] = 0.75
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, 253:256, 253:256, -1].flatten()
        expected_slice = np.array([0.2458, 0.2576, 0.3124, 0.2679, 0.2669, 0.2796, 0.2872, 0.2975, 0.2661])
        assert np.abs(expected_slice - image_slice).max() < 3e-3

    def test_stable_diffusion_simple_inpaint_ddim(self):
        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        pipe.vae = vae
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images

        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3296, 0.4041, 0.4097, 0.4145, 0.4342, 0.4152, 0.4927, 0.4931, 0.4430])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_download_local(self):
        vae = AsymmetricAutoencoderKL.from_pretrained(
            "cross-attention/asymmetric-autoencoder-kl-x-1-5", torch_dtype=torch.float16
        )
        filename = hf_hub_download("runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt")

        pipe = StableDiffusionInpaintPipeline.from_single_file(filename, torch_dtype=torch.float16)
        pipe.vae = vae
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 1
        image_out = pipe(**inputs).images[0]

        assert image_out.shape == (512, 512, 3)


@nightly
@require_torch_gpu
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):
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
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3


class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):
    def test_pil_inputs(self):
        height, width = 32, 32
        im = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        mask = np.random.randint(0, 255, (height, width), dtype=np.uint8) > 127.5
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        t_mask, t_masked, t_image = prepare_mask_and_masked_image(im, mask, height, width, return_image=True)

        self.assertTrue(isinstance(t_mask, torch.Tensor))
        self.assertTrue(isinstance(t_masked, torch.Tensor))
        self.assertTrue(isinstance(t_image, torch.Tensor))

        self.assertEqual(t_mask.ndim, 4)
        self.assertEqual(t_masked.ndim, 4)
        self.assertEqual(t_image.ndim, 4)

        self.assertEqual(t_mask.shape, (1, 1, height, width))
        self.assertEqual(t_masked.shape, (1, 3, height, width))
        self.assertEqual(t_image.shape, (1, 3, height, width))

        self.assertTrue(t_mask.dtype == torch.float32)
        self.assertTrue(t_masked.dtype == torch.float32)
        self.assertTrue(t_image.dtype == torch.float32)

        self.assertTrue(t_mask.min() >= 0.0)
        self.assertTrue(t_mask.max() <= 1.0)
        self.assertTrue(t_masked.min() >= -1.0)
        self.assertTrue(t_masked.min() <= 1.0)
        self.assertTrue(t_image.min() >= -1.0)
        self.assertTrue(t_image.min() >= -1.0)

        self.assertTrue(t_mask.sum() > 0.0)

    def test_np_inputs(self):
        height, width = 32, 32

        im_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im_pil = Image.fromarray(im_np)
        mask_np = (
            np.random.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            > 127.5
        )
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        t_mask_pil, t_masked_pil, t_image_pil = prepare_mask_and_masked_image(
            im_pil, mask_pil, height, width, return_image=True
        )

        self.assertTrue((t_mask_np == t_mask_pil).all())
        self.assertTrue((t_masked_np == t_masked_pil).all())
        self.assertTrue((t_image_np == t_image_pil).all())

    def test_torch_3D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_3D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_4D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_batch_4D_3D(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    2,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])
        t_image_np = torch.cat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_batch_4D_4D(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    2,
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy()[0] for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])
        t_image_np = torch.cat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_shape_mismatch(self):
        height, width = 32, 32

        # test height and width
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    3,
                    height,
                    width,
                ),
                torch.randn(64, 64),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    2,
                    3,
                    height,
                    width,
                ),
                torch.randn(4, 64, 64),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    2,
                    3,
                    height,
                    width,
                ),
                torch.randn(4, 1, 64, 64),
                height,
                width,
                return_image=True,
            )

    def test_type_mismatch(self):
        height, width = 32, 32

        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                height,
                width,
                return_image=True,
            )
        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                torch.rand(
                    3,
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )

    def test_channels_first(self):
        height, width = 32, 32

        # test channels first for 3D tensors
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.rand(height, width, 3),
                torch.rand(
                    3,
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )

    def test_tensor_range(self):
        height, width = 32, 32

        # test im <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.ones(
                    3,
                    height,
                    width,
                )
                * 2,
                torch.rand(
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )
        # test im >= -1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.ones(
                    3,
                    height,
                    width,
                )
                * (-2),
                torch.rand(
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )
        # test mask <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.ones(
                    height,
                    width,
                )
                * 2,
                height,
                width,
                return_image=True,
            )
        # test mask >= 0
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.ones(
                    height,
                    width,
                )
                * -1,
                height,
                width,
                return_image=True,
            )