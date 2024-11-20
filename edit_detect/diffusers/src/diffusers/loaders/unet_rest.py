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
import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from torch import nn

from ..models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    is_accelerate_available,
    is_torch_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .unet_loader_utils import _maybe_expand_lora_scales
from .utils import AttnProcsLayers


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)


TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    @validate_hf_hub_args
            # Unsafe code />















            state_dict = {format_to_lora_compatible(k): v for k, v in state_dict.items()}

            if network_alphas is not None:
                network_alphas = {format_to_lora_compatible(k): v for k, v in network_alphas.items()}
        return state_dict, network_alphas

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import torch
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        ```
        """
        from ..models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_0,
            CustomDiffusionXFormersAttnProcessor,
        )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:


            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(
                x,
                (CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_0, CustomDiffusionXFormersAttnProcessor),
            )
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(
                        x,
                        (
                            CustomDiffusionAttnProcessor,
                            CustomDiffusionAttnProcessor2_0,
                            CustomDiffusionXFormersAttnProcessor,
                        ),
                    )
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_fuse_lora"):
                module._fuse_lora(self.lora_scale, self._safe_fusing)

            if adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported in your environment. Please switch"
                    " to PEFT backend to use this argument by installing latest PEFT and transformers."
                    " `pip install -U peft transformers`"
                )
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer

            merge_kwargs = {"safe_merge": self._safe_fusing}

            if isinstance(module, BaseTunerLayer):
                if self.lora_scale != 1.0:
                    module.scale_layer(self.lora_scale)

                # For BC with prevous PEFT versions, we need to check the signature
                # of the `merge` method to see if it supports the `adapter_names` argument.
                supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
                if "adapter_names" in supported_merge_kwargs:
                    merge_kwargs["adapter_names"] = adapter_names
                elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                    raise ValueError(
                        "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                        " to the latest version of PEFT. `pip install -U peft`"
                    )

                module.merge(**merge_kwargs)

    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_unfuse_lora"):
                module._unfuse_lora()
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        weights = _maybe_expand_lora_scales(self, weights)

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=False):
        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        updated_state_dict = {}
        image_projection = None
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            with init_context():
                image_projection = ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            with init_context():
                image_projection = IPAdapterFullImageProjection(
                    cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        elif "perceiver_resampler.proj_in.weight" in state_dict:
            # IP-Adapter Face ID Plus
            id_embeddings_dim = state_dict["proj.0.weight"].shape[1]
            embed_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[0]
            hidden_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[1]
            output_dims = state_dict["perceiver_resampler.proj_out.weight"].shape[0]
            heads = state_dict["perceiver_resampler.layers.0.0.to_q.weight"].shape[0] // 64

            with init_context():
                image_projection = IPAdapterFaceIDPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    id_embeddings_dim=id_embeddings_dim,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("perceiver_resampler.", "")
                diffusers_name = diffusers_name.replace("0.to", "attn.to")
                diffusers_name = diffusers_name.replace("0.1.0.", "0.ff.0.")
                diffusers_name = diffusers_name.replace("0.1.1.weight", "0.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("0.1.3.weight", "0.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("1.1.0.", "1.ff.0.")
                diffusers_name = diffusers_name.replace("1.1.1.weight", "1.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.1.3.weight", "1.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("2.1.0.", "2.ff.0.")
                diffusers_name = diffusers_name.replace("2.1.1.weight", "2.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("2.1.3.weight", "2.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("3.1.0.", "3.ff.0.")
                diffusers_name = diffusers_name.replace("3.1.1.weight", "3.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("3.1.3.weight", "3.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("layers.0.0", "layers.0.ln0")
                diffusers_name = diffusers_name.replace("layers.0.1", "layers.0.ln1")
                diffusers_name = diffusers_name.replace("layers.1.0", "layers.1.ln0")
                diffusers_name = diffusers_name.replace("layers.1.1", "layers.1.ln1")
                diffusers_name = diffusers_name.replace("layers.2.0", "layers.2.ln0")
                diffusers_name = diffusers_name.replace("layers.2.1", "layers.2.ln1")
                diffusers_name = diffusers_name.replace("layers.3.0", "layers.3.ln0")
                diffusers_name = diffusers_name.replace("layers.3.1", "layers.3.ln1")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                elif "proj.0.weight" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.weight"] = value
                elif "proj.0.bias" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.bias"] = value
                elif "proj.2.weight" == diffusers_name:
                    updated_state_dict["proj.net.2.weight"] = value
                elif "proj.2.bias" == diffusers_name:
                    updated_state_dict["proj.net.2.bias"] = value
                else:
                    updated_state_dict[diffusers_name] = value

        elif "norm.weight" in state_dict:
            # IP-Adapter Face ID
            id_embeddings_dim_in = state_dict["proj.0.weight"].shape[1]
            id_embeddings_dim_out = state_dict["proj.0.weight"].shape[0]
            multiplier = id_embeddings_dim_out // id_embeddings_dim_in
            norm_layer = "norm.weight"
            cross_attention_dim = state_dict[norm_layer].shape[0]
            num_tokens = state_dict["proj.2.weight"].shape[0] // cross_attention_dim

            with init_context():
                image_projection = IPAdapterFaceIDImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=id_embeddings_dim_in,
                    mult=multiplier,
                    num_tokens=num_tokens,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

            with init_context():
                image_projection = IPAdapterPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    num_queries=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")
                diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
                diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
                diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    updated_state_dict[diffusers_name] = value

        if not low_cpu_mem_usage:
            image_projection.load_state_dict(updated_state_dict)
        else:
            load_model_dict_into_meta(image_projection, updated_state_dict, device=self.device, dtype=self.dtype)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):
        from ..models.attention_processor import (
            AttnProcessor,
            AttnProcessor2_0,
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0,
        )

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = (
                    AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                )
                attn_procs[name] = attn_processor_class()

            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
                )
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

                key_id += 2

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"

        self.to(dtype=self.dtype, device=self.device)

    def _load_ip_adapter_loras(self, state_dicts):
        lora_dicts = {}
        for key_id, name in enumerate(self.attn_processors.keys()):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )
        return lora_dicts