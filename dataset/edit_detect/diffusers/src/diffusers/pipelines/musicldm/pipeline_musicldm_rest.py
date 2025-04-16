# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    ClapTextModelWithProjection,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
)

from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline, StableDiffusionMixin


if is_librosa_available():
    import librosa

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import MusicLDMPipeline
        >>> import torch
        >>> import scipy

        >>> repo_id = "ucsd-reach/musicldm"
        >>> pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        >>> audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

        >>> # save the audio sample as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
        ```
"""


class MusicLDMPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-audio generation using MusicLDM.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.ClapModel`]):
            Frozen text-audio embedding model (`ClapTextModel`), specifically the
            [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant.
        tokenizer ([`PreTrainedTokenizer`]):
            A [`~transformers.RobertaTokenizer`] to tokenize text.
        feature_extractor ([`~transformers.ClapFeatureExtractor`]):
            Feature extractor to compute mel-spectrograms from audio waveforms.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`~transformers.SpeechT5HifiGan`]):
            Vocoder of class `SpeechT5HifiGan`.
    """



    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform

    # Copied from diffusers.pipelines.audioldm2.pipeline_audioldm2.AudioLDM2Pipeline.score_waveforms

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.check_inputs

    # Copied from diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.prepare_latents


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)