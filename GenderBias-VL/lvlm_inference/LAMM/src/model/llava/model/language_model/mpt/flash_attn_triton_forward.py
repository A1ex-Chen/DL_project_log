@staticmethod
def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
    """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
    q, k, v = [(x if x.stride(-1) == 1 else x.contiguous()) for x in [q, k, v]]
    o, lse, ctx.softmax_scale = _flash_attn_forward(q, k, v, bias=bias,
        causal=causal, softmax_scale=softmax_scale)
    ctx.save_for_backward(q, k, v, o, lse, bias)
    ctx.causal = causal
    return o
