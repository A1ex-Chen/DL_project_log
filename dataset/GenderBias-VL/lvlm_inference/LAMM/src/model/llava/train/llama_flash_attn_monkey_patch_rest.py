from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input




# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask

