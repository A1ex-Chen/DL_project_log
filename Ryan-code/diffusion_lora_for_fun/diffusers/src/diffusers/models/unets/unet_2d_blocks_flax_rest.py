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

import flax.linen as nn
import jax.numpy as jnp

from ..attention_flax import FlaxTransformer2DModel
from ..resnet_flax import FlaxDownsample2D, FlaxResnetBlock2D, FlaxUpsample2D


class FlaxCrossAttnDownBlock2D(nn.Module):
    r"""
    Cross Attention 2D Downsizing block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    num_attention_heads: int = 1
    add_downsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1




class FlaxDownBlock2D(nn.Module):
    r"""
    Flax 2D downsizing block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32




class FlaxCrossAttnUpBlock2D(nn.Module):
    r"""
    Cross Attention 2D Upsampling block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsampling layer before each final output
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    num_attention_heads: int = 1
    add_upsample: bool = True
    use_linear_projection: bool = False
    only_cross_attention: bool = False
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1




class FlaxUpBlock2D(nn.Module):
    r"""
    Flax 2D upsampling block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        prev_output_channel (:obj:`int`):
            Output channels from the previous block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int
    prev_output_channel: int
    dropout: float = 0.0
    num_layers: int = 1
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32




class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        num_attention_heads (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    num_attention_heads: int = 1
    use_linear_projection: bool = False
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32
    transformer_layers_per_block: int = 1
