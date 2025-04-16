def scale_values(im):
    scale = 255.0 / (hi - lo)
    offset = -lo * scale
    im = tf.cast(im, tf.float32) * scale + offset
    im = tf.clip_by_value(im, 0.0, 255.0)
    return tf.cast(im, tf.uint8)
