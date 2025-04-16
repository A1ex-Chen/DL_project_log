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

import flax.linen as nn
import jax.numpy as jnp




class FlaxTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
                Parameters `dtype`
    """

    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact


class FlaxTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

    Args:
        dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
    """

    dim: int = 32
    flip_sin_to_cos: bool = False
    freq_shift: float = 1

    @nn.compact