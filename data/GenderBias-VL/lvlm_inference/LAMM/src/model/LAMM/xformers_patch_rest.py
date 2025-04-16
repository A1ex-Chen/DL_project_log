"""
Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
"""

import logging
import math
from typing import Optional, Tuple

import torch
from .modeling_llama import (
    apply_rotary_pos_emb,
    LlamaModel,
    LlamaAttention,
)
from torch import nn

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")



