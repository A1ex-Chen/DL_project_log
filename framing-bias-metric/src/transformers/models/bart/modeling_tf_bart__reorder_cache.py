@staticmethod
def _reorder_cache(past, beam_idx):
    assert len(past) == 2
    encoder_out, decoder_cached_states = past
    reordered_past = []
    for layer_past in decoder_cached_states:
        layer_past_new = {attn_key: _reorder_buffer(attn_cache, beam_idx) for
            attn_key, attn_cache in layer_past.items()}
        reordered_past.append(layer_past_new)
    past = encoder_out, reordered_past
    return past
