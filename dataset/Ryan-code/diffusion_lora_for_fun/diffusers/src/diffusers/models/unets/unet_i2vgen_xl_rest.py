# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import UNet2DConditionLoadersMixin
from ...utils import logging
from ..activations import get_activation
from ..attention import Attention, FeedForward
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..transformers.transformer_temporal import TransformerTemporalModel
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .unet_3d_condition import UNet3DConditionOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class I2VGenXLTransformerTemporalEncoder(nn.Module):



class I2VGenXLUNet(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    I2VGenXL UNet. It is a conditional 3D UNet model that takes a noisy sample, conditional state, and a timestep and
    returns a sample-shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 64): Attention head dim.
        num_attention_heads (`int`, *optional*): The number of attention heads.
    """

    _supports_gradient_checkpointing = False

    @register_to_config

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel._set_gradient_checkpointing

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections


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

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1


        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

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

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel._set_gradient_checkpointing
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu
    def enable_freeu(self, s1, s2, b1, b2):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        fps: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[torch.Tensor]]:
        r"""
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            fps (`torch.Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`torch.Tensor`): Image encodings from the VAE.
            image_embeddings (`torch.Tensor`):
                Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        batch_size, channels, num_frames, height, width = sample.shape

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass `timesteps` as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        t_emb = self.time_embedding(t_emb, timestep_cond)

        # 2. FPS
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps = fps.expand(fps.shape[0])
        fps_emb = self.fps_embedding(self.time_proj(fps).to(dtype=self.dtype))

        # 3. time + FPS embeddings.
        emb = t_emb + fps_emb
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        # 4. context embeddings.
        # The context embeddings consist of both text embeddings from the input prompt
        # AND the image embeddings from the input image. For images, both VAE encodings
        # and the CLIP image embeddings are incorporated.
        # So the final `context_embeddings` becomes the query for cross-attention.
        context_emb = sample.new_zeros(batch_size, 0, self.config.cross_attention_dim)
        context_emb = torch.cat([context_emb, encoder_hidden_states], dim=1)

        image_latents_for_context_embds = image_latents[:, :, :1, :]
        image_latents_context_embs = image_latents_for_context_embds.permute(0, 2, 1, 3, 4).reshape(
            image_latents_for_context_embds.shape[0] * image_latents_for_context_embds.shape[2],
            image_latents_for_context_embds.shape[1],
            image_latents_for_context_embds.shape[3],
            image_latents_for_context_embds.shape[4],
        )
        image_latents_context_embs = self.image_latents_context_embedding(image_latents_context_embs)

        _batch_size, _channels, _height, _width = image_latents_context_embs.shape
        image_latents_context_embs = image_latents_context_embs.permute(0, 2, 3, 1).reshape(
            _batch_size, _height * _width, _channels
        )
        context_emb = torch.cat([context_emb, image_latents_context_embs], dim=1)

        image_emb = self.context_embedding(image_embeddings)
        image_emb = image_emb.view(-1, self.config.in_channels, self.config.cross_attention_dim)
        context_emb = torch.cat([context_emb, image_emb], dim=1)
        context_emb = context_emb.repeat_interleave(repeats=num_frames, dim=0)

        image_latents = image_latents.permute(0, 2, 1, 3, 4).reshape(
            image_latents.shape[0] * image_latents.shape[2],
            image_latents.shape[1],
            image_latents.shape[3],
            image_latents.shape[4],
        )
        image_latents = self.image_latents_proj_in(image_latents)
        image_latents = (
            image_latents[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * height * width, num_frames, channels)
        )
        image_latents = self.image_latents_temporal_encoder(image_latents)
        image_latents = image_latents.reshape(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1, 2)

        # 5. pre-process
        sample = torch.cat([sample, image_latents], dim=1)
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)
        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # 6. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=context_emb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        # 7. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=context_emb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # 8. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context_emb,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 9. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)