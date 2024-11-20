import gc
import unittest

import numpy as np
import torch
from torch.backends.cuda import sdp_kernel

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_2,
    require_torch_gpu,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class ConsistencyModelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ConsistencyModelPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    # Override required_optional_params to remove num_images_per_prompt
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    @property

    @property








@nightly
@require_torch_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):






    @require_torch_2

    @require_torch_2