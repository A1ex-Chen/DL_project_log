import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    PIAPipeline,
    UNet2DConditionModel,
    UNetMotionModel,
)
from diffusers.utils import is_xformers_available, logging
from diffusers.utils.testing_utils import floats_tensor, torch_device

from ..test_pipelines_common import IPAdapterTesterMixin, PipelineFromPipeTesterMixin, PipelineTesterMixin




class PIAPipelineFastTests(IPAdapterTesterMixin, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase):
    pipeline_class = PIAPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
            "cross_attention_kwargs",
        ]
    )
    batch_params = frozenset(["prompt", "image", "generator"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )






    @unittest.skip("Attention slicing is not enabled in this pipeline")


    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")




    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )