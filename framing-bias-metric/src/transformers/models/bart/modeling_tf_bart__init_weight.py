@staticmethod
def _init_weight(n_pos, dim):
    """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for
        j in range(dim)] for pos in range(n_pos)])
    position_enc[:, 0:dim // 2] = np.sin(position_enc[:, 0::2])
    position_enc[:, dim // 2:] = np.cos(position_enc[:, 1::2])
    table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
    tf.stop_gradient(table)
    return table
