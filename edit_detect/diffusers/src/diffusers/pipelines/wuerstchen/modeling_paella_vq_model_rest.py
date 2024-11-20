# Copyright (c) 2022 Dominic Rampas MIT License
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

from typing import Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.autoencoders.vae import DecoderOutput, VectorQuantizer
from ...models.modeling_utils import ModelMixin
from ...models.vq_model import VQEncoderOutput
from ...utils.accelerate_utils import apply_forward_hook


class MixingResidualBlock(nn.Module):
    """
    Residual block with mixing used by Paella's VQ-VAE.
    """




class PaellaVQModel(ModelMixin, ConfigMixin):
    r"""VQ-VAE model from Paella model.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        up_down_scale_factor (int, *optional*, defaults to 2): Up and Downscale factor of the input image.
        levels  (int, *optional*, defaults to 2): Number of levels in the model.
        bottleneck_blocks (int, *optional*, defaults to 12): Number of bottleneck blocks in the model.
        embed_dim (int, *optional*, defaults to 384): Number of hidden channels in the model.
        latent_channels (int, *optional*, defaults to 4): Number of latent channels in the VQ-VAE model.
        num_vq_embeddings (int, *optional*, defaults to 8192): Number of codebook vectors in the VQ-VAE.
        scale_factor (float, *optional*, defaults to 0.3764): Scaling factor of the latent space.
    """

    @register_to_config

    @apply_forward_hook

    @apply_forward_hook
