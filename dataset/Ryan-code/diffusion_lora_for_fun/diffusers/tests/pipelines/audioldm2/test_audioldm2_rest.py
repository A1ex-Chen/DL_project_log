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
import unittest

import numpy as np
import torch
from transformers import (
    ClapAudioConfig,
    ClapConfig,
    ClapFeatureExtractor,
    ClapModel,
    ClapTextConfig,
    GPT2Config,
    GPT2Model,
    RobertaTokenizer,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

from diffusers import (
    AudioLDM2Pipeline,
    AudioLDM2ProjectionModel,
    AudioLDM2UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.testing_utils import enable_full_determinism, nightly, torch_device

from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AudioLDM2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AudioLDM2Pipeline
    params = TEXT_TO_AUDIO_PARAMS
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )











    @unittest.skip("Raises a not implemented error in AudioLDM2")








@nightly
class AudioLDM2PipelineSlowTests(unittest.TestCase):






