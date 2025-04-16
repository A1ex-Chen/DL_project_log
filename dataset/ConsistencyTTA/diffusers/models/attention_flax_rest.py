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

import flax.linen as nn
import jax.numpy as jnp


class FlaxAttention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32






class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32




class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32




class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32




class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

