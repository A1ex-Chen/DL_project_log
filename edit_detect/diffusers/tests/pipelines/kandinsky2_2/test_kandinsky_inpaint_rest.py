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

from diffusers import (
    DDIMScheduler,
    KandinskyV22InpaintPipeline,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
    VQModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    is_flaky,
    load_image,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Dummies:
    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property




class KandinskyV22InpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22InpaintPipeline
    params = ["image_embeds", "negative_image_embeds", "image", "mask_image"]
    batch_params = [
        "image_embeds",
        "negative_image_embeds",
        "image",
        "mask_image",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False
    callback_cfg_params = ["image_embeds", "masked_image", "mask_image"]






    @is_flaky()



    # override default test because we need to zero out mask too in order to make sure final latent is all zero


@slow
@require_torch_gpu
class KandinskyV22InpaintPipelineIntegrationTests(unittest.TestCase):



        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


@slow
@require_torch_gpu
class KandinskyV22InpaintPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_kandinsky_inpaint(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinskyv22/kandinskyv22_inpaint_cat_with_hat_fp16.npy"
        )

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
        )
        mask = np.zeros((768, 768), dtype=np.float32)
        mask[:250, 250:-250] = 1

        prompt = "a hat"

        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        )
        pipe_prior.to(torch_device)

        pipeline = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt,
            generator=generator,
            num_inference_steps=2,
            negative_prompt="",
        ).to_tuple()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipeline(
            image=init_image,
            mask_image=mask,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            generator=generator,
            num_inference_steps=2,
            height=768,
            width=768,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        max_diff = numpy_cosine_similarity_distance(expected_image.flatten(), image.flatten())
        assert max_diff < 1e-4