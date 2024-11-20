import gc
import random
import tempfile
import unittest

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)
from diffusers.utils import is_accelerate_available, is_accelerate_version, load_image, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    floats_tensor,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()




class StableVideoDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableVideoDiffusionPipeline
    params = frozenset(["image"])
    batch_params = frozenset(["image", "generator"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
        ]
    )



    @unittest.skip("Deprecated functionality")

    @unittest.skip("Batched inference works and outputs look correct, but the test is failing")

    @unittest.skip("Test is similar to test_inference_batch_single_identical")



    @unittest.skip("Test is currently failing")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")



    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")


    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.17.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.17.0` or higher",
    )

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )



@slow
@require_torch_gpu
class StableVideoDiffusionPipelineSlowTests(unittest.TestCase):

