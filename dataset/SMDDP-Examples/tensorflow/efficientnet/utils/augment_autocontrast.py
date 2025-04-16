def autocontrast(image: tf.Tensor) ->tf.Tensor:
    """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

    def scale_channel(image: tf.Tensor) ->tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)
        result = tf.cond(hi > lo, lambda : scale_values(image), lambda : image)
        return result
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image
