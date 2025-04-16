@staticmethod
def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length,
    num_chunks_before):
    start_position = (previous_hidden_states.shape[1] // chunk_length -
        num_chunks_before) * chunk_length
    return previous_hidden_states[:, start_position:]
