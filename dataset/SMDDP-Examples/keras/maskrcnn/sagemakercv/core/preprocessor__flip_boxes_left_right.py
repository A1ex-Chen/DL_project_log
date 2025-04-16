def _flip_boxes_left_right(boxes):
    """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1
        )
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
    return flipped_boxes
