def _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal
    =False, softmax_scale=None):
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads
        )
    _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(2),
        o.stride(1), do.stride(0), do.stride(2), do.stride(1), nheads,
        seqlen_q, seqlen_q_rounded, d, BLOCK_M=128, BLOCK_HEADDIM=BLOCK_HEADDIM
        )
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
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
    grid = lambda META: (triton.cdiv(seqlen_k, META['BLOCK_N']) if META[
        'SEQUENCE_PARALLEL'] else 1, batch * nheads)
    _bwd_kernel[grid](q, k, v, bias, do, dq_accum, dk, dv, lse, delta,
        softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0),
        k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *
        bias_strides, do.stride(0), do.stride(2), do.stride(1), dq_accum.
        stride(0), dq_accum.stride(2), dq_accum.stride(1), dk.stride(0), dk
        .stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, 
        seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM)
    dq.copy_(dq_accum)
