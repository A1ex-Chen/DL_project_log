def filter_boxes(boxes, scores, image_shape, min_size_threshold):
    """Filter and remove boxes that are too small or fall outside the image.

  Args:
    boxes: a tensor whose last dimension is 4 representing the
      coordinates of boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
      representing the original scores of the boxes.
    image_shape: a tensor whose shape is the same as, or `broadcastable` to
      `boxes` except the last dimension, which is 2, representing
      [height, width] of the scaled image.
    min_size_threshold: a float representing the minimal box size in each
      side (w.r.t. the scaled image). Boxes whose sides are smaller than it will
      be filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with 0.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the positinon of the filtered boxes filled with 0.
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('filter_boxes'):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height = image_shape[..., 0]
            width = image_shape[..., 1]
        ymin = boxes[..., 0]
        xmin = boxes[..., 1]
        ymax = boxes[..., 2]
        xmax = boxes[..., 3]
        h = ymax - ymin + 1.0
        w = xmax - xmin + 1.0
        yc = ymin + 0.5 * h
        xc = xmin + 0.5 * w
        min_size = tf.cast(tf.maximum(min_size_threshold, 1.0), dtype=boxes
            .dtype)
        filtered_size_mask = tf.logical_and(tf.greater(h, min_size), tf.
            greater(w, min_size))
        filtered_center_mask = tf.logical_and(tf.logical_and(tf.greater(yc,
            0.0), tf.less(yc, height)), tf.logical_and(tf.greater(xc, 0.0),
            tf.less(xc, width)))
        filtered_mask = tf.logical_and(filtered_size_mask, filtered_center_mask
            )
        filtered_scores = tf.where(filtered_mask, scores, tf.zeros_like(scores)
            )
        filtered_boxes = tf.cast(tf.expand_dims(filtered_mask, axis=-1),
            dtype=boxes.dtype) * boxes
        return filtered_boxes, filtered_scores
