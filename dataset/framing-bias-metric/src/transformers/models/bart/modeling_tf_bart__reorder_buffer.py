def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = tf.gather(input_buffer_k, new_order, axis=0)
    return attn_cache
