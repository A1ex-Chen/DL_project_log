# Copyright 2024 Salesforce.com, inc.
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
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import CLIPPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPEncoder




# This is a modified version of the CLIPTextModel from transformers.models.clip.modeling_clip
# Which allows for an extra input of "context embeddings", which are the query embeddings used in Qformer
# They pass through the clip model, along with the text embeddings, and interact with them using self attention
class ContextCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]




class ContextCLIPTextTransformer(nn.Module):




class ContextCLIPTextEmbeddings(nn.Module):
