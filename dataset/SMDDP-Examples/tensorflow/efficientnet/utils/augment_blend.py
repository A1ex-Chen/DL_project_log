def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) ->tf.Tensor:
    """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)
    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)
    difference = image2 - image1
    scaled = factor * difference
    temp = tf.cast(image1, tf.float32) + scaled
    if factor > 0.0 and factor < 1.0:
        return tf.cast(temp, tf.uint8)
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)
