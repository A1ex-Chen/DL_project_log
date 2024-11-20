def normalize_images(features: tf.Tensor, mean_rgb: Tuple[float, ...]=
    MEAN_RGB, stddev_rgb: Tuple[float, ...]=STDDEV_RGB, num_channels: int=3,
    dtype: tf.dtypes.DType=tf.float32, data_format: Text='channels_last'
    ) ->tf.Tensor:
    """Normalizes the input image channels with the given mean and stddev.

  Args:
    features: `Tensor` representing decoded images in float format.
    mean_rgb: the mean of the channels to subtract.
    stddev_rgb: the stddev of the channels to divide.
    num_channels: the number of channels in the input image tensor.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    data_format: the format of the input image tensor
                 ['channels_first', 'channels_last'].

  Returns:
    A normalized image `Tensor`.
  """
    if data_format == 'channels_first':
        stats_shape = [num_channels, 1, 1]
    else:
        stats_shape = [1, 1, num_channels]
    if dtype is not None:
        features = tf.image.convert_image_dtype(features, dtype=dtype)
    if mean_rgb is not None:
        mean_rgb = tf.constant(mean_rgb, shape=stats_shape, dtype=features.
            dtype)
        mean_rgb = tf.broadcast_to(mean_rgb, tf.shape(features))
        features = features - mean_rgb
    if stddev_rgb is not None:
        stddev_rgb = tf.constant(stddev_rgb, shape=stats_shape, dtype=
            features.dtype)
        stddev_rgb = tf.broadcast_to(stddev_rgb, tf.shape(features))
        features = features / stddev_rgb
    return features
