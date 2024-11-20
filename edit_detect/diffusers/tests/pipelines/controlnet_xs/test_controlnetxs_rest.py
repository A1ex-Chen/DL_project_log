# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import traceback
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    ControlNetXSAdapter,
    DDIMScheduler,
    LCMScheduler,
    StableDiffusionControlNetXSPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
    load_numpy,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ...models.autoencoders.test_models_vae import (
    get_asym_autoencoder_kl_config,
    get_autoencoder_kl_config,
    get_autoencoder_tiny_config,
    get_consistency_vae_config,
)
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
)


enable_full_determinism()




# Will be run via run_test_in_subprocess


class ControlNetXSPipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionControlNetXSPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    test_attention_slicing = False



    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )





    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")


@slow
@require_torch_gpu
class ControlNetXSPipelineSlowTests(unittest.TestCase):



    @require_python39_or_higher
    @require_torch_2