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

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from safetensors import safe_open

from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
from .unet_loader_utils import _maybe_expand_lora_scales


if is_transformers_available():
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    from ..models.attention_processor import (
        AttnProcessor,
        AttnProcessor2_0,
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

logger = logging.get_logger(__name__)


class IPAdapterMixin:
    """Mixin for handling IP Adapters."""

    @validate_hf_hub_args

