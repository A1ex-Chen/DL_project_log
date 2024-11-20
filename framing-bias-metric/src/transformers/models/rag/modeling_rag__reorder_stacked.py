def _reorder_stacked(hidden_states):
    n_docs = hidden_states.shape[0] // beam_idx.shape[0]
    hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
    hidden_states = hidden_states.index_select(0, beam_idx)
    return hidden_states.view(-1, *hidden_states.shape[2:])
