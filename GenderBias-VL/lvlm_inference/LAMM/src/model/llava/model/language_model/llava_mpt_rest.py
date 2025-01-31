#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import math

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel
from model.llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMPTConfig(MPTConfig):
    model_type = "llava_mpt"


class LlavaMPTModel(LlavaMetaModel, MPTModel):
    config_class = LlavaMPTConfig

    


class LlavaMPTForCausalLM(MPTForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMPTConfig
    supports_gradient_checkpointing = True







AutoConfig.register("llava_mpt", LlavaMPTConfig)
AutoModelForCausalLM.register(LlavaMPTConfig, LlavaMPTForCausalLM)