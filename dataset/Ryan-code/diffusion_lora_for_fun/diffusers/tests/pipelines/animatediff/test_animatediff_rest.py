import gc
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AnimateDiffPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
    UNetMotionModel,
)
from diffusers.utils import is_xformers_available, logging
from diffusers.utils.testing_utils import numpy_cosine_similarity_distance, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineFromPipeTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
)




class AnimateDiffPipelineFastTests(
    IPAdapterTesterMixin, SDFunctionTesterMixin, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase
):
    pipeline_class = AnimateDiffPipeline
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




    @unittest.skip("Attention slicing is not enabled in this pipeline")




    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")




    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )



@slow
@require_torch_gpu
class AnimateDiffPipelineSlowTests(unittest.TestCase):

