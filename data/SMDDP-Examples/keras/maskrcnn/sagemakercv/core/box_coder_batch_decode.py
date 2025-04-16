def batch_decode(encoded_boxes, box_coder, anchors):
    """Decode a batch of encoded boxes.

  This op takes a batch of encoded bounding boxes and transforms
  them to a batch of bounding boxes specified by their corners in
  the order of [y_min, x_min, y_max, x_max].

  Args:
    encoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      code_size] representing the location of the objects.
    box_coder: a BoxCoder object.
    anchors: a BoxList of anchors used to encode `encoded_boxes`.

  Returns:
    decoded_boxes: a float32 tensor of shape [batch_size, num_anchors,
      coder_size] representing the corners of the objects in the order
      of [y_min, x_min, y_max, x_max].

  Raises:
    ValueError: if batch sizes of the inputs are inconsistent, or if
    the number of anchors inferred from encoded_boxes and anchors are
    inconsistent.
  """
    if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
        raise ValueError(
            'The number of anchors inferred from encoded_boxes and anchors are inconsistent: shape[1] of encoded_boxes %s should be equal to the number of anchors: %s.'
             % (encoded_boxes.get_shape()[1].value, anchors.num_boxes_static())
            )
    decoded_boxes = tf.stack([box_coder.decode(boxes, anchors).get() for
        boxes in tf.unstack(encoded_boxes)])
    return decoded_boxes
