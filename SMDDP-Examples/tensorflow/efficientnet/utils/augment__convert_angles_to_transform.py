def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
    image_height: tf.Tensor) ->tf.Tensor:
    """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).

  Raises:
    `TypeError` if `angles` is not rank 0 or 1.

  """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError('Angles should have a rank 0 or 1.')
    x_offset = (image_width - 1 - (tf.math.cos(angles) * (image_width - 1) -
        tf.math.sin(angles) * (image_height - 1))) / 2.0
    y_offset = (image_height - 1 - (tf.math.sin(angles) * (image_width - 1) +
        tf.math.cos(angles) * (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(values=[tf.math.cos(angles)[:, None], -tf.math.sin(
        angles)[:, None], x_offset[:, None], tf.math.sin(angles)[:, None],
        tf.math.cos(angles)[:, None], y_offset[:, None], tf.zeros((
        num_angles, 2), tf.dtypes.float32)], axis=1)
