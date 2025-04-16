def rotate(image: tf.Tensor, degrees: float) ->tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.

  """
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)
    original_ndims = tf.rank(image)
    image = to_4d(image)
    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)
    transforms = _convert_angles_to_transform(angles=radians, image_width=
        image_width, image_height=image_height)
    image = transform(image, transforms=transforms)
    return from_4d(image, original_ndims)
