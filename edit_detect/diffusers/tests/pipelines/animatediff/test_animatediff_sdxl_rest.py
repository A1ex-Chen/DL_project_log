import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import diffusers
from diffusers import (
    AnimateDiffSDXLPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
    UNetMotionModel,
)
from diffusers.utils import is_xformers_available, logging
from diffusers.utils.testing_utils import torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
    SDXLOptionalComponentsTesterMixin,
)




class AnimateDiffPipelineSDXLFastTests(
    IPAdapterTesterMixin,
    SDFunctionTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
    unittest.TestCase,
):
    pipeline_class = AnimateDiffSDXLPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
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
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})




    @unittest.skip("Attention slicing is not enabled in this pipeline")


    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")




    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )