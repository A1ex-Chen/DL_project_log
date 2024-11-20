"""Attention layers."""
import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch import nn
from .norm import LPLayerNorm






class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, low_precision_layernorm: bool=False, verbose: int=0, device: Optional[str]=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.chunk(3, dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context, attn_weights, past_key_value) = self.attn_fn(query, key, value, self.n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
        return (self.out_proj(context), attn_weights, past_key_value)

class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, low_precision_layernorm: bool=False, verbose: int=0, device: Optional[str]=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(d_model, device=device)
            self.k_ln = layernorm_class(self.head_dim, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.split([self.d_model, self.head_dim, self.head_dim], dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        (context, attn_weights, past_key_value) = self.attn_fn(query, key, value, self.n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights, multiquery=True)
        return (self.out_proj(context), attn_weights, past_key_value)






class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """



def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal, use_sequence_id):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi=False, alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            (device, dtype) = (attn_bias.device, attn_bias.dtype)
            attn_bias = attn_bias.add(build_alibi_bias(n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, device=device, dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=8, device=None, dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)
ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention, 'multiquery_attention': MultiQueryAttention}