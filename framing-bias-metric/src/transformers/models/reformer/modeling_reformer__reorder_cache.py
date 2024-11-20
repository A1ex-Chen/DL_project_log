def _reorder_cache(self, past, beam_idx):
    reord_past_buckets_states = []
    for layer_past in past:
        if layer_past[0] is not None:
            reord_buckets = layer_past[0].index_select(0, beam_idx)
        else:
            reord_buckets = None
        reord_hidden_states = layer_past[1].index_select(0, beam_idx)
        reord_past_buckets_states.append((reord_buckets, reord_hidden_states))
    return reord_past_buckets_states
