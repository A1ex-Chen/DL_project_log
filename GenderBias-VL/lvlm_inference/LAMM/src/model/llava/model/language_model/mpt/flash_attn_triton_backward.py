@staticmethod
def backward(ctx, do):
    q, k, v, o, lse, bias = ctx.saved_tensors
    assert not ctx.needs_input_grad[3
        ], 'FlashAttention does not support bias gradient yet'
    with torch.inference_mode():
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=bias,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale)
    return dq, dk, dv, None, None, None
