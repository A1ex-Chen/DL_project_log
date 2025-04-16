def build_lut(histo, step):
    lut = (tf.cumsum(histo) + step // 2) // step
    lut = tf.concat([[0], lut[:-1]], 0)
    return tf.clip_by_value(lut, 0, 255)
