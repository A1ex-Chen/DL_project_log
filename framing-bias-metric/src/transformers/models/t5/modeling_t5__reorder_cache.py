def _reorder_cache(self, past, beam_idx):
    if past is None:
        logger.warning(
            'You might want to consider setting `use_cache=True` to speed up decoding'
            )
        return past
    reordered_decoder_past = ()
    for layer_past_states in past:
        reordered_layer_past_states = ()
        for layer_past_state in layer_past_states:
            reordered_layer_past_states = reordered_layer_past_states + (
                layer_past_state.index_select(0, beam_idx),)
        assert reordered_layer_past_states[0].shape == layer_past_states[0
            ].shape
        assert len(reordered_layer_past_states) == len(layer_past_states)
        reordered_decoder_past = reordered_decoder_past + (
            reordered_layer_past_states,)
    return reordered_decoder_past
