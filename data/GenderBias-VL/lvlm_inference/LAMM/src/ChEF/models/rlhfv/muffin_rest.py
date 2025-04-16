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


from typing import List, Optional, Tuple, Union

import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import timm
from .beit3 import beit3_large_patch16_448
from timm.models.layers import trunc_normal_
from .utils import build_transform


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)


class Beit3LlavaConfig(LlamaConfig):
    model_type = "beit3_llava"


class Beit3LlavaLlamaModel(LlamaModel):
    config_class = Beit3LlavaConfig





class Beit3LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = Beit3LlavaConfig





AutoConfig.register("beit3_llava", Beit3LlavaConfig)
AutoModelForCausalLM.register(Beit3LlavaConfig, Beit3LlavaLlamaForCausalLM)