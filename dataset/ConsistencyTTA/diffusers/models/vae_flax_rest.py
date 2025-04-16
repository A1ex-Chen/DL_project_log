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

# JAX implementation of VQGAN from taming-transformers https://github.com/CompVis/taming-transformers

import math
from functools import partial
from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import ConfigMixin, flax_register_to_config
from ..utils import BaseOutput
from .modeling_flax_utils import FlaxModelMixin


@flax.struct.dataclass
class FlaxDecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    sample: jnp.ndarray


@flax.struct.dataclass
class FlaxAutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`FlaxDiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `FlaxDiagonalGaussianDistribution`.
            `FlaxDiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "FlaxDiagonalGaussianDistribution"


class FlaxUpsample2D(nn.Module):
    """
    Flax implementation of 2D Upsample layer

    Args:
        in_channels (`int`):
            Input channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    dtype: jnp.dtype = jnp.float32




class FlaxDownsample2D(nn.Module):
    """
    Flax implementation of 2D Downsample layer

    Args:
        in_channels (`int`):
            Input channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    dtype: jnp.dtype = jnp.float32




class FlaxResnetBlock2D(nn.Module):
    """
    Flax implementation of 2D Resnet Block.

    Args:
        in_channels (`int`):
            Input channels
        out_channels (`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for group norm.
        use_nin_shortcut (:obj:`bool`, *optional*, defaults to `None`):
            Whether to use `nin_shortcut`. This activates a new layer inside ResNet block
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int = None
    dropout: float = 0.0
    groups: int = 32
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32




class FlaxAttentionBlock(nn.Module):
    r"""
    Flax Convolutional based multi-head attention block for diffusion-based VAE.

    Parameters:
        channels (:obj:`int`):
            Input channels
        num_head_channels (:obj:`int`, *optional*, defaults to `None`):
            Number of attention heads
        num_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for group norm
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`

    """
    channels: int
    num_head_channels: int = None
    num_groups: int = 32
    dtype: jnp.dtype = jnp.float32





class FlaxDownEncoderBlock2D(nn.Module):
    r"""
    Flax Resnet blocks-based Encoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsample layer
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32




class FlaxUpDecoderBlock2D(nn.Module):
    r"""
    Flax Resnet blocks-based Decoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsample layer
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32




class FlaxUNetMidBlock2D(nn.Module):
    r"""
    Flax Unet Mid-Block module.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet and Attention block group norm
        attn_num_head_channels (:obj:`int`, *optional*, defaults to `1`):
            Number of attention heads for each attention block
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    attn_num_head_channels: int = 1
    dtype: jnp.dtype = jnp.float32




class FlaxEncoder(nn.Module):
    r"""
    Flax Implementation of VAE Encoder.

    This model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        down_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            DownEncoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    double_z: bool = False
    dtype: jnp.dtype = jnp.float32




class FlaxDecoder(nn.Module):
    r"""
    Flax Implementation of VAE Decoder.

    This model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        up_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            UpDecoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            parameters `dtype`
    """
    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: int = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    dtype: jnp.dtype = jnp.float32




class FlaxDiagonalGaussianDistribution(object):






@flax_register_to_config
class FlaxAutoencoderKL(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    Flax Implementation of Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational
    Bayes by Diederik P. Kingma and Max Welling.

    This model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        down_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            DownEncoder block type
        up_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            UpDecoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        latent_channels (:obj:`int`, *optional*, defaults to `4`):
            Latent space channels
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            Norm num group
        sample_size (:obj:`int`, *optional*, defaults to 32):
            Sample input size
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            parameters `dtype`
    """
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    dtype: jnp.dtype = jnp.float32




