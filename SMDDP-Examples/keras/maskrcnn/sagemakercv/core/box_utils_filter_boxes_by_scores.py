def filter_boxes_by_scores(boxes, scores, min_score_threshold):
    """Filter and remove boxes whose scores are smaller than the threshold.

  Args:
    boxes: a tensor whose last dimension is 4 representing the
      coordinates of boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
      representing the original scores of the boxes.
    min_score_threshold: a float representing the minimal box score threshold.
      Boxes whose score are smaller than it will be filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with 0.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('filter_boxes_by_scores'):
        filtered_mask = tf.greater(scores, min_score_threshold)
        filtered_scores = tf.where(filtered_mask, scores, tf.zeros_like(scores)
            )
        filtered_boxes = tf.cast(tf.expand_dims(filtered_mask, axis=-1),
            dtype=boxes.dtype) * boxes
        return filtered_boxes, filtered_scores
