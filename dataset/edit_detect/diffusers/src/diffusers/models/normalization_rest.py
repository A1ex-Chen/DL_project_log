# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_torch_version
from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """




class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """




class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """




class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """




class AdaLayerNormContinuous(nn.Module):



if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):



class GlobalResponseNorm(nn.Module):
    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105




class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class GlobalResponseNorm(nn.Module):
    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x