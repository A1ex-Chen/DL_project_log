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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .modeling_utils import ModelMixin
from .vae import Decoder, DecoderOutput, Encoder, VectorQuantizer


@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Encoded output sample of the model. Output of the last layer of the model.
    """

    latents: torch.FloatTensor


class VQModel(ModelMixin, ConfigMixin):
    r"""VQ-VAE model from the paper Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals and Koray
    Kavukcuoglu.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    @register_to_config


