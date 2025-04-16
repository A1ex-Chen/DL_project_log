def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False,
    alibi=False, alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(build_alibi_bias(n_heads, seq_len,
                full=not causal, alibi_bias_max=alibi_bias_max, device=
                device, dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
