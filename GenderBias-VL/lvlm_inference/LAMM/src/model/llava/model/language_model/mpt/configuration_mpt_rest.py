"""A HuggingFace-style model configuration."""
from typing import Dict, Optional, Union
from transformers import PretrainedConfig
attn_config_defaults: Dict = {'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}
init_config_defaults: Dict = {'name': 'kaiming_normal_', 'fan_mode': 'fan_in', 'init_nonlinearity': 'relu', 'init_div_is_residual': True, 'emb_init_std': None, 'emb_init_uniform_lim': None, 'init_std': None, 'init_gain': 0.0}

class MPTConfig(PretrainedConfig):
    model_type = 'mpt'


