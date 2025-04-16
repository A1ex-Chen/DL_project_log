""" 
This code is based on https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py.
"""

from typing import List, Optional, Tuple, Dict

import logging
import torch
import transformers
from .modeling_llama import (
    apply_rotary_pos_emb,
    LlamaModel,
    LlamaAttention,
)

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input







# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask


  