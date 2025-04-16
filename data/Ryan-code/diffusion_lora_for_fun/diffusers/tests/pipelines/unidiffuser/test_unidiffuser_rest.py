import gc
import random
import traceback
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    GPT2Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    nightly,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


# Will be run via run_test_in_subprocess


class UniDiffuserPipelineFastTests(
    PipelineTesterMixin, PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, unittest.TestCase
):
    pipeline_class = UniDiffuserPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    # vae_latents, not latents, is the argument that corresponds to VAE latent inputs
    image_latents_params = frozenset(["vae_latents"])




















    @require_torch_gpu

    @require_torch_gpu

    @require_torch_gpu


@nightly
@require_torch_gpu
class UniDiffuserPipelineSlowTests(unittest.TestCase):







    @unittest.skip(reason="Skip torch.compile test to speed up the slow test suite.")
    @require_torch_2


@nightly
@require_torch_gpu
class UniDiffuserPipelineNightlyTests(unittest.TestCase):





