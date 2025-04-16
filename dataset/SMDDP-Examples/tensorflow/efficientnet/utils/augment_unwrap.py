def unwrap(image: tf.Tensor, replace: int) ->tf.Tensor:
    """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
    image_shape = tf.shape(image)
    flattened_image = tf.reshape(image, [-1, image_shape[2]])
    alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)
    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)
    flattened_image = tf.where(tf.equal(alpha_channel, 0), tf.ones_like(
        flattened_image, dtype=image.dtype) * replace, flattened_image)
    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image
