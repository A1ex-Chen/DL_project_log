# Copyright 2022 The Music Spectrogram Diffusion Authors.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ...models import T5FilmDecoder
from ...schedulers import DDPMScheduler
from ...utils import is_onnx_available, logging, randn_tensor


if is_onnx_available():
    from ..onnx_utils import OnnxRuntimeModel

from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .continous_encoder import SpectrogramContEncoder
from .notes_encoder import SpectrogramNotesEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TARGET_FEATURE_LENGTH = 256


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    _optional_components = ["melgan"]






    @torch.no_grad()