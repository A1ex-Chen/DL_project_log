def scaled_multihead_dot_product_attention(query, key, value, n_heads,
    past_key_value=None, softmax_scale=None, attn_bias=None,
    key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False,
    needs_weights=False, multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = k, v
    b, _, s_q, d = q.shape
    s_k = k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1
            ) != s_k or attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q:
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.'
                )
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn(
                'Propogating key_padding_mask to the attention module ' +
                'and applying it within the attention module can cause ' +
                'unneccessary computation/memory usage. Consider integrating '
                 +
                'into attn_bias once and passing that to each attention ' +
                'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1,
            1, s_k)), min_val)
    if is_causal and not q.size(2) == 1:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q,
            s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p,
            training=training, inplace=True)
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value
