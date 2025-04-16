import gc
import inspect
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LatentConsistencyModelPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import IPAdapterTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LatentConsistencyModelPipelineFastTests(
    IPAdapterTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = LatentConsistencyModelPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"negative_prompt", "negative_prompt_embeds"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {"negative_prompt"}
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS








    # skip because lcm pipeline apply cfg differently

    # override default test because the final latent variable is "denoised" instead of "latents"


@slow
@require_torch_gpu
class LatentConsistencyModelPipelineSlowTests(unittest.TestCase):




        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


@slow
@require_torch_gpu
class LatentConsistencyModelPipelineSlowTests(unittest.TestCase):
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

    def test_lcm_onestep(self):
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", safety_checker=None)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1025, 0.0911, 0.0984, 0.0981, 0.0901, 0.0918, 0.1055, 0.0940, 0.0730])
        assert np.abs(image_slice - expected_slice).max() < 1e-3

    def test_lcm_multistep(self):
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", safety_checker=None)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.01855, 0.01855, 0.01489, 0.01392, 0.01782, 0.01465, 0.01831, 0.02539, 0.0])
        assert np.abs(image_slice - expected_slice).max() < 1e-3