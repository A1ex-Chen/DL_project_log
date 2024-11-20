def filter_boxes_2(boxes, scores, min_size, height, width, scale):
    """Filter out boxes that are too small.
    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates
        of boxes in ymin, xmin, ymax, xmax order.
      scores: a tensor such as all but the last dimensions are the same as
        `boxes`. The last dimension is 1. It represents the scores.
      min_size: an integer specifying the minimal size.
      height: an integer, a scalar or a tensor such as all but the last dimensions
        are the same as `boxes`. The last dimension is 1. It represents the height
        of the image.
      width: an integer, a scalar or a tensor such as all but the last dimensions
        are the same as `boxes`. The last dimension is 1. It represents the width
        of the image.
      scale: an integer, a scalar or a tensor such as all but the last dimensions
        are the same as `boxes`. The last dimension is 1. It represents the scale
        of the image.
    Returns:
      filtered_boxes: a tensor whose shape is the same as `boxes` representing the
        filtered boxes.
      filtered_scores: a tensor whose shape is the same as `scores` representing
        the filtered scores.
    """
    with tf.name_scope('filter_box'):
        y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)
        h = y_max - y_min + 1.0
        w = x_max - x_min + 1.0
        yc = y_min + h / 2.0
        xc = x_min + w / 2.0
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
        scale = tf.cast(scale, dtype=boxes.dtype)
        min_size = tf.cast(tf.maximum(min_size, 1), dtype=boxes.dtype)
        size_mask = tf.logical_and(tf.greater_equal(h, min_size * scale),
            tf.greater_equal(w, min_size * scale))
        center_mask = tf.logical_and(tf.less(yc, height), tf.less(xc, width))
        selected_mask = tf.logical_and(size_mask, center_mask)
        filtered_scores = tf.where(selected_mask, scores, tf.zeros_like(scores)
            )
        filtered_boxes = tf.cast(selected_mask, dtype=boxes.dtype) * boxes
    return filtered_boxes, filtered_scores
