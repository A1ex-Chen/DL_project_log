def compute_outer_boxes(boxes, image_shape, scale=1.0):
    """Compute outer box encloses an object with a margin.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
    scale: a float number specifying the scale of output outer boxes to input
      `boxes`.

  Returns:
    outer_boxes: a tensor whose shape is the same as `boxes` representing the
      outer boxes.
  """
    if scale < 1.0:
        raise ValueError(
            'scale is {}, but outer box scale must be greater than 1.0.'.
            format(scale))
    centers_y = (boxes[..., 0] + boxes[..., 2]) / 2.0
    centers_x = (boxes[..., 1] + boxes[..., 3]) / 2.0
    box_height = (boxes[..., 2] - boxes[..., 0]) * scale
    box_width = (boxes[..., 3] - boxes[..., 1]) * scale
    outer_boxes = tf.stack([centers_y - box_height / 2.0, centers_x - 
        box_width / 2.0, centers_y + box_height / 2.0, centers_x + 
        box_width / 2.0], axis=1)
    outer_boxes = clip_boxes(outer_boxes, image_shape)
    return outer_boxes
