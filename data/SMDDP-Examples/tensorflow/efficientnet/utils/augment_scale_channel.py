def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
        lut = (tf.cumsum(histo) + step // 2) // step
        lut = tf.concat([[0], lut[:-1]], 0)
        return tf.clip_by_value(lut, 0, 255)
    result = tf.cond(tf.equal(step, 0), lambda : im, lambda : tf.gather(
        build_lut(histo, step), im))
    return tf.cast(result, tf.uint8)
