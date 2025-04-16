@staticmethod
def _reorder_cache(past, beam_idx):
    reordered_past = []
    for layer_past in past:
        layer_past_new = {attn_key: _reorder_buffer(attn_cache, beam_idx) for
            attn_key, attn_cache in layer_past.items()}
        reordered_past.append(layer_past_new)
    return reordered_past
