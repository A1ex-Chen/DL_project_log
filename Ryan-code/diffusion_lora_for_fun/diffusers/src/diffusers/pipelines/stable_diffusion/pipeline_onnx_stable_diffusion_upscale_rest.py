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
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from ...utils import deprecate, logging
from ..onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput


logger = logging.get_logger(__name__)




class OnnxStableDiffusionUpscalePipeline(DiffusionPipeline):
    vae: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    low_res_scheduler: DDPMScheduler
    scheduler: KarrasDiffusionSchedulers
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPImageProcessor

    _optional_components = ["safety_checker", "feature_extractor"]
    _is_onnx = True





