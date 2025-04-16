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
import math
from typing import Optional, Tuple

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention_processor import Attention
from ..embeddings import get_timestep_embedding
from ..modeling_utils import ModelMixin


class T5FilmDecoder(ModelMixin, ConfigMixin):
    r"""
    T5 style decoder with FiLM conditioning.

    Args:
        input_dims (`int`, *optional*, defaults to `128`):
            The number of input dimensions.
        targets_length (`int`, *optional*, defaults to `256`):
            The length of the targets.
        d_model (`int`, *optional*, defaults to `768`):
            Size of the input hidden states.
        num_layers (`int`, *optional*, defaults to `12`):
            The number of `DecoderLayer`'s to use.
        num_heads (`int`, *optional*, defaults to `12`):
            The number of attention heads to use.
        d_kv (`int`, *optional*, defaults to `64`):
            Size of the key-value projection vectors.
        d_ff (`int`, *optional*, defaults to `2048`):
            The number of dimensions in the intermediate feed-forward layer of `DecoderLayer`'s.
        dropout_rate (`float`, *optional*, defaults to `0.1`):
            Dropout probability.
    """

    @register_to_config




class DecoderLayer(nn.Module):
    r"""
    T5 decoder layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`, *optional*, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """




class T5LayerSelfAttentionCond(nn.Module):
    r"""
    T5 style self-attention layer with conditioning.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
    """




class T5LayerCrossAttention(nn.Module):
    r"""
    T5 style cross-attention layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """




class T5LayerFFCond(nn.Module):
    r"""
    T5 style feed-forward conditional layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """




class T5DenseGatedActDense(nn.Module):
    r"""
    T5 style feed-forward layer with gated activations and dropout.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
    """




class T5LayerNorm(nn.Module):
    r"""
    T5 style layer normalization module.

    Args:
        hidden_size (`int`):
            Size of the input hidden states.
        eps (`float`, `optional`, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """




class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """



class T5FiLMLayer(nn.Module):
    """
    T5 style FiLM Layer.

    Args:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
    """

