def decode_and_center_crop(image_bytes: tf.Tensor, image_size: int=
    IMAGE_SIZE, crop_padding: int=CROP_PADDING) ->tf.Tensor:
    """Crops to center of image with padding then scales image_size.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    crop_padding: the padding size to use when centering the crop.

  Returns:
    A decoded and cropped image `Tensor`.
  """
    decoded = image_bytes.dtype != tf.string
    shape = tf.shape(image_bytes) if decoded else tf.image.extract_jpeg_shape(
        image_bytes)
    image_height = shape[0]
    image_width = shape[1]
    padded_center_crop_size = tf.cast(image_size / (image_size +
        crop_padding) * tf.cast(tf.minimum(image_height, image_width), tf.
        float32), tf.int32)
    offset_height = (image_height - padded_center_crop_size + 1) // 2
    offset_width = (image_width - padded_center_crop_size + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
        padded_center_crop_size, padded_center_crop_size])
    if decoded:
        image = tf.image.crop_to_bounding_box(image_bytes, offset_height=
            offset_height, offset_width=offset_width, target_height=
            padded_center_crop_size, target_width=padded_center_crop_size)
    else:
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window,
            channels=3)
    image = resize_image(image_bytes=image, height=image_size, width=image_size
        )
    return image
