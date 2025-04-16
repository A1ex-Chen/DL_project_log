def normalize_boxes(boxes, image_shape):
    """Converts boxes to the normalized coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    normalized_boxes: a tensor whose shape is the same as `boxes` representing
      the normalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('normalize_boxes'):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height = image_shape[..., 0:1]
            width = image_shape[..., 1:2]
        ymin = boxes[..., 0:1] / height
        xmin = boxes[..., 1:2] / width
        ymax = boxes[..., 2:3] / height
        xmax = boxes[..., 3:4] / width
        normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
        return normalized_boxes
