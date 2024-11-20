def denormalize_boxes(boxes, image_shape):
    """Converts boxes normalized by [height, width] to pixel coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    denormalized_boxes: a tensor whose shape is the same as `boxes` representing
      the denormalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
    with tf.name_scope('denormalize_boxes'):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height, width = tf.split(image_shape, 2, axis=-1)
        ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
        ymin = ymin * height
        xmin = xmin * width
        ymax = ymax * height
        xmax = xmax * width
        denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
        return denormalized_boxes
