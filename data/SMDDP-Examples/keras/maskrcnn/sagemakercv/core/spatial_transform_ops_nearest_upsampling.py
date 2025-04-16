def nearest_upsampling(data, scale):
    """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    data_up: A tensor with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
    with tf.name_scope('nearest_upsampling'):
        bs, h, w, c = tf.unstack(tf.shape(data))
        output = tf.stack([data] * scale, axis=3)
        output = tf.stack([output] * scale, axis=2)
        return tf.reshape(output, [bs, h * scale, w * scale, c])
    return tf.reshape(data, [bs, h * scale, w * scale, c])
