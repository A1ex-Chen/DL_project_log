import gc
import unittest

import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    PriorTransformer,
    StableUnCLIPPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.utils.testing_utils import enable_full_determinism, load_numpy, nightly, require_torch_gpu, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)


enable_full_determinism()


class StableUnCLIPPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableUnCLIPPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    # TODO(will) Expected attn_bias.stride(1) == 0 to be true, but got false
    test_xformers_attention = False



    # Overriding PipelineTesterMixin::test_attention_slicing_forward_pass
    # because UnCLIP GPU undeterminism requires a looser check.

    # Overriding PipelineTesterMixin::test_inference_batch_single_identical
    # because UnCLIP undeterminism requires a looser check.


@nightly
@require_torch_gpu
class StableUnCLIPPipelineIntegrationTests(unittest.TestCase):


