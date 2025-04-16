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

import tempfile
import unittest

import numpy as np

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    OnnxStableDiffusionPipeline,
    PNDMScheduler,
)
from diffusers.utils.testing_utils import is_onnx_available, nightly, require_onnxruntime, require_torch_gpu

from ..test_pipelines_onnx_common import OnnxPipelineTesterMixin


if is_onnx_available():
    import onnxruntime as ort


class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
    hub_checkpoint = "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline"











@nightly
@require_onnxruntime
@require_torch_gpu
class OnnxStableDiffusionPipelineIntegrationTests(unittest.TestCase):
    @property

    @property






        test_callback_fn.has_been_called = False

        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="onnx",
            safety_checker=None,
            feature_extractor=None,
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "Andromeda galaxy in a bottle"

        generator = np.random.RandomState(0)
        pipe(
            prompt=prompt,
            num_inference_steps=5,
            guidance_scale=7.5,
            generator=generator,
            callback=test_callback_fn,
            callback_steps=1,
        )
        assert test_callback_fn.has_been_called
        assert number_of_steps == 6

    def test_stable_diffusion_no_safety_checker(self):
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="onnx",
            safety_checker=None,
            feature_extractor=None,
            provider=self.gpu_provider,
            sess_options=self.gpu_options,
        )
        assert isinstance(pipe, OnnxStableDiffusionPipeline)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

        # check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = OnnxStableDiffusionPipeline.from_pretrained(tmpdirname)

        # sanity check that the pipeline still works
        assert pipe.safety_checker is None
        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None