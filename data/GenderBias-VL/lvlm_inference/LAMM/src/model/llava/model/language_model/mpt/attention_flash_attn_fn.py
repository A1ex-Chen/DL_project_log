def flash_attn_fn(query, key, value, n_heads, past_key_value=None,
    softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=
    False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False
    ):
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0')
    check_valid_inputs(query, key, value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = key, value
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')
    batch_size, seqlen = query.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1):]
    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = (bert_padding.
        unpad_input(query, query_padding_mask))
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(key,
        key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=1 if
        multiquery else n_heads)
    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=1 if
        multiquery else n_heads)
    if multiquery:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.
            size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads,
            value_unpad.size(-1))
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(query_unpad,
        key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, softmax_scale=softmax_scale, causal=
        reset_is_causal, return_attn_probs=needs_weights)
    output = bert_padding.pad_input(rearrange(output_unpad,
        'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return output, None, past_key_value
