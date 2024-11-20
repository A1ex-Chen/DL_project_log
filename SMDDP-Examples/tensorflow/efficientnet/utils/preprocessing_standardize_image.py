def standardize_image(image_bytes: tf.Tensor, stddev: Tuple[float, ...],
    num_channels: int=3, dtype: tf.dtypes.DType=tf.float32) ->tf.Tensor:
    """Divides the given stddev from each image channel.

  For example:
    stddev = [123.68, 116.779, 103.939]
    image_bytes = standardize_image(image_bytes, stddev)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    stddev: a C-vector of values to divide from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `stddev`.
  """
    if image_bytes.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    if len(stddev) != num_channels:
        raise ValueError('len(stddev) must match the number of channels')
    stddev = tf.broadcast_to(stddev, tf.shape(image_bytes))
    if dtype is not None:
        stddev = tf.cast(stddev, dtype=dtype)
    return image_bytes / stddev
