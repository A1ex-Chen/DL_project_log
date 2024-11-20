def _reorder_buffer(attn_cache):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = _reorder_stacked(input_buffer_k)
    return attn_cache
