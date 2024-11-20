def preprocess_for_train(image_bytes: tf.Tensor, image_size: int=IMAGE_SIZE,
    augmenter: Optional[augment.ImageAugment]=None, mean_subtract: bool=
    False, standardize: bool=False, dtype: tf.dtypes.DType=tf.float32
    ) ->tf.Tensor:
    """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of
      arbitrary size of dtype tf.uint8.
    image_size: image height/width dimension.
    augmenter: the image augmenter to apply.
    mean_subtract: whether or not to apply mean subtraction.
    standardize: whether or not to apply standardization.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
    images = decode_crop_and_flip(image_bytes=image_bytes)
    images = resize_image(images, height=image_size, width=image_size)
    if mean_subtract:
        images = mean_image_subtraction(image_bytes=images, means=MEAN_RGB)
    if standardize:
        images = standardize_image(image_bytes=images, stddev=STDDEV_RGB)
    if augmenter is not None:
        images = augmenter.distort(images)
    if dtype is not None:
        images = tf.image.convert_image_dtype(images, dtype)
    return images
