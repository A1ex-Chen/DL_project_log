def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for
        j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))
