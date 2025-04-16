def _decode_png_mask(png_bytes):
    mask = tf.squeeze(tf.io.decode_png(png_bytes, channels=1, dtype=tf.
        uint8), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask.set_shape([None, None])
    return mask
