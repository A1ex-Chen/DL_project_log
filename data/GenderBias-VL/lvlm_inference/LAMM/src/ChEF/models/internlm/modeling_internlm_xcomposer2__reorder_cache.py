@staticmethod
def _reorder_cache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += tuple(past_state.index_select(0, beam_idx.to(
            past_state.device)) for past_state in layer_past),
    return reordered_past
