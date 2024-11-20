def preprocess_for_eval(image_bytes: tf.Tensor, image_size: int=IMAGE_SIZE,
    num_channels: int=3, mean_subtract: bool=False, standardize: bool=False,
    dtype: tf.dtypes.DType=tf.float32) ->tf.Tensor:
    """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    num_channels: number of image input channels.
    mean_subtract: whether or not to apply mean subtraction.
    standardize: whether or not to apply standardization.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
    images = decode_and_center_crop(image_bytes, image_size)
    images = tf.reshape(images, [image_size, image_size, num_channels])
    if mean_subtract:
        images = mean_image_subtraction(image_bytes=images, means=MEAN_RGB)
    if standardize:
        images = standardize_image(image_bytes=images, stddev=STDDEV_RGB)
    if dtype is not None:
        images = tf.image.convert_image_dtype(images, dtype=dtype)
    return images
