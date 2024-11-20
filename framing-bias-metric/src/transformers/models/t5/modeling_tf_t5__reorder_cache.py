def _reorder_cache(self, past, beam_idx) ->Tuple:
    if len(past) < 2:
        logger.warning(
            'You might want to consider setting `use_cache=True` to speed up decoding'
            )
        return past
    decoder_past = past[1]
    past = past[0],
    reordered_decoder_past = ()
    for layer_past_states in decoder_past:
        reordered_layer_past_states = ()
        for layer_past_state in layer_past_states:
            reordered_layer_past_states = reordered_layer_past_states + (tf
                .gather(layer_past_state, beam_idx),)
        assert shape_list(reordered_layer_past_states[0]) == shape_list(
            layer_past_states[0])
        assert len(reordered_layer_past_states) == len(layer_past_states)
        reordered_decoder_past = reordered_decoder_past + (
            reordered_layer_past_states,)
    return past + (reordered_decoder_past,)
