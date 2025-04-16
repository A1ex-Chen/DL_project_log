def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal,
    use_sequence_id):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return 1, n_heads, seq_len, seq_len
            return 1, n_heads, 1, seq_len
        elif prefix_lm or use_sequence_id:
            return 1, 1, seq_len, seq_len
        return None
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
