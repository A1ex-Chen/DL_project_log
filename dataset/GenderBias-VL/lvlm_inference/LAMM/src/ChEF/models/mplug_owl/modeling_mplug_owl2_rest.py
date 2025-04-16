#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mplug_owl2 import MPLUGOwl2Config, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig, MPLUGOwl2QwenConfig
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel
from .modeling_llama2 import replace_llama_modality_adaptive
from .modeling_qwen import QWenLMHeadModel, QWenModel
from .constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from icecream import ic
from transformers.modeling_utils import PreTrainedModel

class MPLUGOwl2MetaModel:
    



class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod





class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config


class MPLUGOwl2QWenModel(MPLUGOwl2MetaModel, QWenModel):
    config_class = MPLUGOwl2QwenConfig


class MPLUGOwl2LlamaForCausalLM(LlamaForCausalLM, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config






class MPLUGOwl2QWenForCausalLM(QWenLMHeadModel, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2QwenConfig




AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)
AutoConfig.register("mplug_owl2_1", MPLUGOwl2QwenConfig)
AutoModelForCausalLM.register(MPLUGOwl2QwenConfig, MPLUGOwl2QWenForCausalLM)

replace_llama_modality_adaptive()

if __name__ == "__main__":
    config = MPLUGOwl2Config.from_pretrained('/cpfs01/shared/public/test/vicuna-7b-v1.5/')
    from icecream import ic
    # config = MPLUGOwl2Config()
    model =  MPLUGOwl2LlamaForCausalLM(config)
    
    images = torch.randn(2, 3, 448, 448)
    input_ids = torch.cat([
        torch.ones(8).long(), torch.tensor([-1]*1).long(), torch.ones(8).long(), torch.tensor([-1]*1).long(), torch.ones(8).long()
    ], dim=0).unsqueeze(0)
    labels = input_ids.clone()
    labels[labels < 0] = -100
    
    # image_feature = model.encode_images(images)
    # ic(image_feature.shape)
    
    output = model(images=images, input_ids=input_ids, labels=labels)
    ic(output.loss)
    ic(output.logits.shape)
    
    model.save_pretrained('/cpfs01/shared/public/test/tmp_owl')