def _reorder_cache(cache_dict, beam_idx):
    for k, key_value_states in cache_dict.items():
        if key_value_states is not None:
            cache_dict[k] = key_value_states.index_select(0, beam_idx)
    return cache_dict
