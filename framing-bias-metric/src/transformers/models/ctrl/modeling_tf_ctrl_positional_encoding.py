def positional_encoding(position, d_model_size):
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(
        d_model_size)[np.newaxis, :], d_model_size)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.cast(np.concatenate([sines, cosines], axis=-1), dtype
        =tf.float32)
    return pos_encoding
