# Copyright 2022 The Music Spectrogram Diffusion Authors.
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

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ....models import T5FilmDecoder
from ....schedulers import DDPMScheduler
from ....utils import is_onnx_available, logging
from ....utils.torch_utils import randn_tensor


if is_onnx_available():
    from ...onnx_utils import OnnxRuntimeModel

from ...pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .continuous_encoder import SpectrogramContEncoder
from .notes_encoder import SpectrogramNotesEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TARGET_FEATURE_LENGTH = 256


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional audio generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        notes_encoder ([`SpectrogramNotesEncoder`]):
        continuous_encoder ([`SpectrogramContEncoder`]):
        decoder ([`T5FilmDecoder`]):
            A [`T5FilmDecoder`] to denoise the encoded audio latents.
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `decoder` to denoise the encoded audio latents.
        melgan ([`OnnxRuntimeModel`]):
    """

    _optional_components = ["melgan"]






    @torch.no_grad()