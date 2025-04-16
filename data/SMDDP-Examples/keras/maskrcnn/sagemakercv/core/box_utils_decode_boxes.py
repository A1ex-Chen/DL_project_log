def decode_boxes(encoded_boxes, anchors, weights=None):
    """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
    if encoded_boxes.shape[-1] != 4:
        raise ValueError('encoded_boxes.shape[-1] is {:d}, but must be 4.'.
            format(encoded_boxes.shape[-1]))
    with tf.name_scope('decode_boxes'):
        encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
        dy = encoded_boxes[..., 0:1]
        dx = encoded_boxes[..., 1:2]
        dh = encoded_boxes[..., 2:3]
        dw = encoded_boxes[..., 3:4]
        if weights:
            dy /= weights[0]
            dx /= weights[1]
            dh /= weights[2]
            dw /= weights[3]
        dh = tf.minimum(dh, BBOX_XFORM_CLIP)
        dw = tf.minimum(dw, BBOX_XFORM_CLIP)
        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]
        anchor_h = anchor_ymax - anchor_ymin + 1.0
        anchor_w = anchor_xmax - anchor_xmin + 1.0
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w
        decoded_boxes_yc = dy * anchor_h + anchor_yc
        decoded_boxes_xc = dx * anchor_w + anchor_xc
        decoded_boxes_h = tf.exp(dh) * anchor_h
        decoded_boxes_w = tf.exp(dw) * anchor_w
        decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
        decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
        decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
        decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0
        decoded_boxes = tf.concat([decoded_boxes_ymin, decoded_boxes_xmin,
            decoded_boxes_ymax, decoded_boxes_xmax], axis=-1)
        return decoded_boxes
