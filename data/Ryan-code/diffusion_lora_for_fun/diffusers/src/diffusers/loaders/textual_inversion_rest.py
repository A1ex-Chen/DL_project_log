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
from typing import Dict, List, Optional, Union

import safetensors
import torch
from huggingface_hub.utils import validate_hf_hub_args
from torch import nn

from ..models.modeling_utils import load_state_dict
from ..utils import _get_model_file, is_accelerate_available, is_transformers_available, logging


if is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

TEXT_INVERSION_NAME = "learned_embeds.bin"
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"


@validate_hf_hub_args


class TextualInversionLoaderMixin:
    r"""
    Load Textual Inversion tokens and embeddings to the tokenizer and text encoder.
    """




    @staticmethod

    @staticmethod

    @validate_hf_hub_args

        # / Unsafe Code >
