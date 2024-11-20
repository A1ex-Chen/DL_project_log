def to_absolute_coordinates(boxes, height, width):
    """Converted normalized box coordinates to absolute ones.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    height: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the height
      of the image.
    width: an integer, a scalar or a tensor such as all but the last dimensions
      are the same as `boxes`. The last dimension is 1. It represents the width
      of the image.

  Returns:
    absolute_boxes: a tensor whose shape is the same as `boxes` representing the
      boxes in absolute coordinates.
  """
    with tf.name_scope('denormalize_box'):
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
        y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)
        y_min = y_min * height
        x_min = x_min * width
        y_max = y_max * height
        x_max = x_max * width
        absolute_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    return absolute_boxes
