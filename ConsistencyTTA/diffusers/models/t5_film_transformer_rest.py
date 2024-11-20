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

import torch
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from .attention_processor import Attention
from .embeddings import get_timestep_embedding
from .modeling_utils import ModelMixin


class T5FilmDecoder(ModelMixin, ConfigMixin):
    @register_to_config




class DecoderLayer(nn.Module):



class T5LayerSelfAttentionCond(nn.Module):



class T5LayerCrossAttention(nn.Module):



class T5LayerFFCond(nn.Module):



class T5DenseGatedActDense(nn.Module):



class T5LayerNorm(nn.Module):



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """



class T5FiLMLayer(nn.Module):
    """
    FiLM Layer
    """

