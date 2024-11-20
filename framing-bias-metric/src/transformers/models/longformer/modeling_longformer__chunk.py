@staticmethod
def _chunk(hidden_states, window_overlap):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
    hidden_states = hidden_states.view(hidden_states.size(0), hidden_states
        .size(1) // (window_overlap * 2), window_overlap * 2, hidden_states
        .size(2))
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1
    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
