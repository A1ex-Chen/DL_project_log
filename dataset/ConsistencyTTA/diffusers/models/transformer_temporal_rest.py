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
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .attention import BasicTransformerBlock
from .modeling_utils import ModelMixin


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`)
            Hidden states conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each TransformerBlock should contain two self-attention layers
    """

    @register_to_config
