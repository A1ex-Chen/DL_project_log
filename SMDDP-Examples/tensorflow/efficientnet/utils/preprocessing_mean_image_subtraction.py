def mean_image_subtraction(image_bytes: tf.Tensor, means: Tuple[float, ...],
    num_channels: int=3, dtype: tf.dtypes.DType=tf.float32) ->tf.Tensor:
    """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image_bytes = mean_image_subtraction(image_bytes, means)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
    if image_bytes.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    means = tf.broadcast_to(means, tf.shape(image_bytes))
    if dtype is not None:
        means = tf.cast(means, dtype=dtype)
    return image_bytes - means
