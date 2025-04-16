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

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..utils import BaseOutput
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from .modeling_flax_utils import FlaxModelMixin
from .unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
)


@flax.struct.dataclass
class FlaxControlNetOutput(BaseOutput):
    """
    The output of [`FlaxControlNetModel`].

    Args:
        down_block_res_samples (`jnp.ndarray`):
        mid_block_res_sample (`jnp.ndarray`):
    """

    down_block_res_samples: jnp.ndarray
    mid_block_res_sample: jnp.ndarray


class FlaxControlNetConditioningEmbedding(nn.Module):
    conditioning_embedding_channels: int
    block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)
    dtype: jnp.dtype = jnp.float32




@flax_register_to_config
class FlaxControlNetModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A ControlNet model.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it’s generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Inherent JAX features such as the following are supported:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        controlnet_conditioning_channel_order (`str`, *optional*, defaults to `rgb`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
    """

    sample_size: int = 32
    in_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    only_cross_attention: Union[bool, Tuple[bool, ...]] = False
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    controlnet_conditioning_channel_order: str = "rgb"
    conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256)


