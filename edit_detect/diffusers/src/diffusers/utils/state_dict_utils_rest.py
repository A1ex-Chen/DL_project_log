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
"""
State dict utilities: utility methods for converting state dicts easily
"""

import enum

from .logging import get_logger


logger = get_logger(__name__)


class StateDictType(enum.Enum):
    """
    The mode to use when converting state dicts.
    """

    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    DIFFUSERS = "diffusers"


# We need to define a proper mapping for Unet since it uses different output keys than text encoder
# e.g. to_q_lora -> q_proj / to_q
UNET_TO_DIFFUSERS = {
    ".to_out_lora.up": ".to_out.0.lora_B",
    ".to_out_lora.down": ".to_out.0.lora_A",
    ".to_q_lora.down": ".to_q.lora_A",
    ".to_q_lora.up": ".to_q.lora_B",
    ".to_k_lora.down": ".to_k.lora_A",
    ".to_k_lora.up": ".to_k.lora_B",
    ".to_v_lora.down": ".to_v.lora_A",
    ".to_v_lora.up": ".to_v.lora_B",
    ".lora.up": ".lora_B",
    ".lora.down": ".lora_A",
    ".to_out.lora_magnitude_vector": ".to_out.0.lora_magnitude_vector",
}


DIFFUSERS_TO_PEFT = {
    ".q_proj.lora_linear_layer.up": ".q_proj.lora_B",
    ".q_proj.lora_linear_layer.down": ".q_proj.lora_A",
    ".k_proj.lora_linear_layer.up": ".k_proj.lora_B",
    ".k_proj.lora_linear_layer.down": ".k_proj.lora_A",
    ".v_proj.lora_linear_layer.up": ".v_proj.lora_B",
    ".v_proj.lora_linear_layer.down": ".v_proj.lora_A",
    ".out_proj.lora_linear_layer.up": ".out_proj.lora_B",
    ".out_proj.lora_linear_layer.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

DIFFUSERS_OLD_TO_PEFT = {
    ".to_q_lora.up": ".q_proj.lora_B",
    ".to_q_lora.down": ".q_proj.lora_A",
    ".to_k_lora.up": ".k_proj.lora_B",
    ".to_k_lora.down": ".k_proj.lora_A",
    ".to_v_lora.up": ".v_proj.lora_B",
    ".to_v_lora.down": ".v_proj.lora_A",
    ".to_out_lora.up": ".out_proj.lora_B",
    ".to_out_lora.down": ".out_proj.lora_A",
    ".lora_linear_layer.up": ".lora_B",
    ".lora_linear_layer.down": ".lora_A",
}

PEFT_TO_DIFFUSERS = {
    ".q_proj.lora_B": ".q_proj.lora_linear_layer.up",
    ".q_proj.lora_A": ".q_proj.lora_linear_layer.down",
    ".k_proj.lora_B": ".k_proj.lora_linear_layer.up",
    ".k_proj.lora_A": ".k_proj.lora_linear_layer.down",
    ".v_proj.lora_B": ".v_proj.lora_linear_layer.up",
    ".v_proj.lora_A": ".v_proj.lora_linear_layer.down",
    ".out_proj.lora_B": ".out_proj.lora_linear_layer.up",
    ".out_proj.lora_A": ".out_proj.lora_linear_layer.down",
    "to_k.lora_A": "to_k.lora.down",
    "to_k.lora_B": "to_k.lora.up",
    "to_q.lora_A": "to_q.lora.down",
    "to_q.lora_B": "to_q.lora.up",
    "to_v.lora_A": "to_v.lora.down",
    "to_v.lora_B": "to_v.lora.up",
    "to_out.0.lora_A": "to_out.0.lora.down",
    "to_out.0.lora_B": "to_out.0.lora.up",
}

DIFFUSERS_OLD_TO_DIFFUSERS = {
    ".to_q_lora.up": ".q_proj.lora_linear_layer.up",
    ".to_q_lora.down": ".q_proj.lora_linear_layer.down",
    ".to_k_lora.up": ".k_proj.lora_linear_layer.up",
    ".to_k_lora.down": ".k_proj.lora_linear_layer.down",
    ".to_v_lora.up": ".v_proj.lora_linear_layer.up",
    ".to_v_lora.down": ".v_proj.lora_linear_layer.down",
    ".to_out_lora.up": ".out_proj.lora_linear_layer.up",
    ".to_out_lora.down": ".out_proj.lora_linear_layer.down",
    ".to_k.lora_magnitude_vector": ".k_proj.lora_magnitude_vector",
    ".to_v.lora_magnitude_vector": ".v_proj.lora_magnitude_vector",
    ".to_q.lora_magnitude_vector": ".q_proj.lora_magnitude_vector",
    ".to_out.lora_magnitude_vector": ".out_proj.lora_magnitude_vector",
}

PEFT_TO_KOHYA_SS = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
    # This is not a comprehensive dict as kohya format requires replacing `.` with `_` in keys,
    # adding prefixes and adding alpha values
    # Check `convert_state_dict_to_kohya` for more
}

PEFT_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_PEFT,
    StateDictType.DIFFUSERS: DIFFUSERS_TO_PEFT,
}

DIFFUSERS_STATE_DICT_MAPPINGS = {
    StateDictType.DIFFUSERS_OLD: DIFFUSERS_OLD_TO_DIFFUSERS,
    StateDictType.PEFT: PEFT_TO_DIFFUSERS,
}

KOHYA_STATE_DICT_MAPPINGS = {StateDictType.PEFT: PEFT_TO_KOHYA_SS}

KEYS_TO_ALWAYS_REPLACE = {
    ".processor.": ".",
}











