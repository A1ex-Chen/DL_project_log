@staticmethod
def _pad_and_diagonalize(chunked_hidden_states):
    """
        shift every row 1 step right, converting columns into diagonals.

        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
    total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(
        chunked_hidden_states)
    paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
    chunked_hidden_states = tf.pad(chunked_hidden_states, paddings)
    chunked_hidden_states = tf.reshape(chunked_hidden_states, (
        total_num_heads, num_chunks, -1))
    chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
    chunked_hidden_states = tf.reshape(chunked_hidden_states, (
        total_num_heads, num_chunks, window_overlap, window_overlap +
        hidden_dim))
    chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    return chunked_hidden_states
