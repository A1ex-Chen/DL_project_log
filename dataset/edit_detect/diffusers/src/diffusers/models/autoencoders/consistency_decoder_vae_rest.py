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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...schedulers import ConsistencyDecoderScheduler
from ...utils import BaseOutput
from ...utils.accelerate_utils import apply_forward_hook
from ...utils.torch_utils import randn_tensor
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..modeling_utils import ModelMixin
from ..unets.unet_2d import UNet2DModel
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder


@dataclass
class ConsistencyDecoderVAEOutput(BaseOutput):
    """
    Output of encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class ConsistencyDecoderVAE(ModelMixin, ConfigMixin):
    r"""
    The consistency decoder used with DALL-E 3.

    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline, ConsistencyDecoderVAE

        >>> vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", vae=vae, torch_dtype=torch.float16
        ... ).to("cuda")

        >>> image = pipe("horse", generator=torch.manual_seed(0)).images[0]
        >>> image
        ```
    """

    @register_to_config

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.enable_tiling

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.disable_tiling

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.enable_slicing

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.disable_slicing

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor

    @apply_forward_hook

    @apply_forward_hook

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_v

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_h



        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )


        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[ConsistencyDecoderVAEOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] instead of a plain
                tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return ConsistencyDecoderVAEOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        num_inference_steps: int = 2,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        Decodes the input latent vector `z` using the consistency decoder VAE model.

        Args:
            z (torch.Tensor): The input latent vector.
            generator (Optional[torch.Generator]): The random number generator. Default is None.
            return_dict (bool): Whether to return the output as a dictionary. Default is True.
            num_inference_steps (int): The number of inference steps. Default is 2.

        Returns:
            Union[DecoderOutput, Tuple[torch.Tensor]]: The decoded output.

        """
        z = (z * self.config.scaling_factor - self.means) / self.stds

        scale_factor = 2 ** (len(self.config.block_out_channels) - 1)
        z = F.interpolate(z, mode="nearest", scale_factor=scale_factor)

        batch_size, _, height, width = z.shape

        self.decoder_scheduler.set_timesteps(num_inference_steps, device=self.device)

        x_t = self.decoder_scheduler.init_noise_sigma * randn_tensor(
            (batch_size, 3, height, width), generator=generator, dtype=z.dtype, device=z.device
        )

        for t in self.decoder_scheduler.timesteps:
            model_input = torch.concat([self.decoder_scheduler.scale_model_input(x_t, t), z], dim=1)
            model_output = self.decoder_unet(model_input, t).sample[:, :3, :, :]
            prev_sample = self.decoder_scheduler.step(model_output, t, x_t, generator).prev_sample
            x_t = prev_sample

        x_0 = x_t

        if not return_dict:
            return (x_0,)

        return DecoderOutput(sample=x_0)

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_v
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    # Copied from diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_h
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[ConsistencyDecoderVAEOutput, Tuple]:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] instead of a
                plain tuple.

        Returns:
            [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] or `tuple`:
                If return_dict is True, a [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return ConsistencyDecoderVAEOutput(latent_dist=posterior)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*, defaults to `None`):
                Generator to use for sampling.

        Returns:
            [`DecoderOutput`] or `tuple`:
                If return_dict is True, a [`DecoderOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, generator=generator).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)