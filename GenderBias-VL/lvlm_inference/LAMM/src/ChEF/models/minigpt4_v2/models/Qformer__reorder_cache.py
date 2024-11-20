def _reorder_cache(self, past, beam_idx):
    reordered_past = ()
    for layer_past in past:
        reordered_past += tuple(past_state.index_select(0, beam_idx) for
            past_state in layer_past),
    return reordered_past
