def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16
        ], 'Only support fp16 and bf16'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError(
                'Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)'
                )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)
        ) if has_bias else (0, 0, 0)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device,
        dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device,
        dtype=torch.float32)
    o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads
        )
    _fwd_kernel[grid](q, k, v, bias, o, lse, tmp, softmax_scale, q.stride(0
        ), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1), *bias_strides, o.stride(0),
        o.stride(2), o.stride(1), nheads, seqlen_q, seqlen_k,
        seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type,
        causal, BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK, num_warps=
        num_warps, num_stages=1)
    return o, lse, softmax_scale
