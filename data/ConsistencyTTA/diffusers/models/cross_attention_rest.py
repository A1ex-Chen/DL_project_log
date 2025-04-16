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
from ..utils import deprecate
from .attention_processor import (  # noqa: F401
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRALinearLayer,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    SlicedAttnProcessor,
    XFormersAttnProcessor,
)
from .attention_processor import AttnProcessor as AttnProcessorRename  # noqa: F401


deprecate(
    "cross_attention",
    "0.18.0",
    "Importing from cross_attention is deprecated. Please import from diffusers.models.attention_processor instead.",
    standard_warn=False,
)


AttnProcessor = AttentionProcessor


class CrossAttention(Attention):


class CrossAttnProcessor(AttnProcessorRename):


class LoRACrossAttnProcessor(LoRAAttnProcessor):


class CrossAttnAddedKVProcessor(AttnAddedKVProcessor):


class XFormersCrossAttnProcessor(XFormersAttnProcessor):


class LoRAXFormersCrossAttnProcessor(LoRAXFormersAttnProcessor):


class SlicedCrossAttnProcessor(SlicedAttnProcessor):


class SlicedCrossAttnAddedKVProcessor(SlicedAttnAddedKVProcessor):