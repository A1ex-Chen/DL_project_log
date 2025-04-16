"""
Copied from https://github.com/HazyResearch/flash-attention/blob/eff9fe6b8076df59d64d7a3f464696738a3c7c24/flash_attn/flash_attn_triton.py
update imports to use 'triton_pre_mlir'

*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""
import math
import torch
import triton_pre_mlir as triton
import triton_pre_mlir.language as tl

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit

@triton.jit

@triton.jit

@triton.jit


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit



class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod

    @staticmethod
flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply

class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod

    @staticmethod
flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply

class FlashAttnFunc(torch.autograd.Function):

    @staticmethod

    @staticmethod
flash_attn_func = FlashAttnFunc.apply