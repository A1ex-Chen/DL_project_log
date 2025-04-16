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
import re
from contextlib import nullcontext
from typing import Optional

from huggingface_hub.utils import validate_hf_hub_args

from ..utils import deprecate, is_accelerate_available, logging
from .single_file_utils import (
    SingleFileComponentError,
    convert_controlnet_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_stable_cascade_unet_single_file_to_diffusers,
    create_controlnet_diffusers_config_from_ldm,
    create_unet_diffusers_config_from_ldm,
    create_vae_diffusers_config_from_ldm,
    fetch_diffusers_config,
    fetch_original_config,
    load_single_file_checkpoint,
)


logger = logging.get_logger(__name__)


if is_accelerate_available():
    from accelerate import init_empty_weights

    from ..models.modeling_utils import load_model_dict_into_meta


SINGLE_FILE_LOADABLE_CLASSES = {
    "StableCascadeUNet": {
        "checkpoint_mapping_fn": convert_stable_cascade_unet_single_file_to_diffusers,
    },
    "UNet2DConditionModel": {
        "checkpoint_mapping_fn": convert_ldm_unet_checkpoint,
        "config_mapping_fn": create_unet_diffusers_config_from_ldm,
        "default_subfolder": "unet",
        "legacy_kwargs": {
            "num_in_channels": "in_channels",  # Legacy kwargs supported by `from_single_file` mapped to new args
        },
    },
    "AutoencoderKL": {
        "checkpoint_mapping_fn": convert_ldm_vae_checkpoint,
        "config_mapping_fn": create_vae_diffusers_config_from_ldm,
        "default_subfolder": "vae",
    },
    "ControlNetModel": {
        "checkpoint_mapping_fn": convert_controlnet_checkpoint,
        "config_mapping_fn": create_controlnet_diffusers_config_from_ldm,
    },
}




class FromOriginalModelMixin:
    """
    Load pretrained weights saved in the `.ckpt` or `.safetensors` format into a model.
    """

    @classmethod
    @validate_hf_hub_args