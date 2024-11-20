def encode_boxes(boxes, anchors, weights=None):
    """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('encode_boxes'):
        boxes = tf.cast(boxes, dtype=anchors.dtype)
        y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)
        box_h = y_max - y_min + 1.0
        box_w = x_max - x_min + 1.0
        box_yc = y_min + 0.5 * box_h
        box_xc = x_min + 0.5 * box_w
        anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = tf.split(anchors,
            4, axis=-1)
        anchor_h = anchor_ymax - anchor_ymin + 1.0
        anchor_w = anchor_xmax - anchor_xmin + 1.0
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w
        encoded_dy = (box_yc - anchor_yc) / anchor_h
        encoded_dx = (box_xc - anchor_xc) / anchor_w
        encoded_dh = tf.math.log(box_h / anchor_h)
        encoded_dw = tf.math.log(box_w / anchor_w)
        if weights:
            encoded_dy *= weights[0]
            encoded_dx *= weights[1]
            encoded_dh *= weights[2]
            encoded_dw *= weights[3]
        encoded_boxes = tf.concat([encoded_dy, encoded_dx, encoded_dh,
            encoded_dw], axis=-1)
    return encoded_boxes
