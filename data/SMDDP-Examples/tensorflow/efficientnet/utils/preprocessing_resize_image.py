def resize_image(image_bytes: tf.Tensor, height: int=IMAGE_SIZE, width: int
    =IMAGE_SIZE) ->tf.Tensor:
    """Resizes an image to a given height and width.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    height: image height dimension.
    width: image width dimension.

  Returns:
    A tensor containing the resized image.

  """
    return tf.compat.v1.image.resize(image_bytes, [height, width], method=
        tf.image.ResizeMethod.BILINEAR, align_corners=False)
