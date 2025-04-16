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
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder, MaskConditionDecoder


class AsymmetricAutoencoderKL(ModelMixin, ConfigMixin):
    r"""
    Designing a Better Asymmetric VQGAN for StableDiffusion https://arxiv.org/abs/2306.04632 . A VAE model with KL loss
    for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        down_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of down block output channels.
        layers_per_down_block (`int`, *optional*, defaults to `1`):
            Number layers for down block.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        up_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of up block output channels.
        layers_per_up_block (`int`, *optional*, defaults to `1`):
            Number layers for up block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        norm_num_groups (`int`, *optional*, defaults to `32`):
            Number of groups to use for the first normalization layer in ResNet blocks.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    @register_to_config

    @apply_forward_hook


    @apply_forward_hook
