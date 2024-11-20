def clip_boxes(boxes, image_shape):
    """Clips boxes to image boundaries.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('clip_boxes'):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height = image_shape[..., 0:1]
            width = image_shape[..., 1:2]
        ymin = boxes[..., 0:1]
        xmin = boxes[..., 1:2]
        ymax = boxes[..., 2:3]
        xmax = boxes[..., 3:4]
        clipped_ymin = tf.maximum(tf.minimum(ymin, height - 1.0), 0.0)
        clipped_ymax = tf.maximum(tf.minimum(ymax, height - 1.0), 0.0)
        clipped_xmin = tf.maximum(tf.minimum(xmin, width - 1.0), 0.0)
        clipped_xmax = tf.maximum(tf.minimum(xmax, width - 1.0), 0.0)
        clipped_boxes = tf.concat([clipped_ymin, clipped_xmin, clipped_ymax,
            clipped_xmax], axis=-1)
        return clipped_boxes
