@staticmethod
def _reorder_cache(past, beam_idx):
    """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

    def _reorder_stacked(hidden_states):
        n_docs = hidden_states.shape[0] // beam_idx.shape[0]
        hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:]
            )
        hidden_states = hidden_states.index_select(0, beam_idx)
        return hidden_states.view(-1, *hidden_states.shape[2:])

    def _reorder_buffer(attn_cache):
        for k, input_buffer_k in attn_cache.items():
            if input_buffer_k is not None:
                attn_cache[k] = _reorder_stacked(input_buffer_k)
        return attn_cache
    reordered_past = []
    for layer_past in past:
        layer_past_new = {attn_key: _reorder_buffer(attn_cache) for 
            attn_key, attn_cache in layer_past.items()}
        reordered_past.append(layer_past_new)
    return reordered_past
