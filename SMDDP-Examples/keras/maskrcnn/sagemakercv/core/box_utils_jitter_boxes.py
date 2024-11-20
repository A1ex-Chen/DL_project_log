def jitter_boxes(boxes, noise_scale=0.025):
    """Jitter the box coordinates by some noise distribution.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    noise_scale: a python float which specifies the magnitude of noise. The
      rule of thumb is to set this between (0, 0.1]. The default value is found
      to mimic the noisy detections best empirically.

  Returns:
    jittered_boxes: a tensor whose shape is the same as `boxes` representing
      the jittered boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
            boxes.shape[-1]))
    with tf.name_scope('jitter_boxes'):
        bbox_jitters = tf.random_normal(boxes.get_shape(), stddev=noise_scale)
        ymin = boxes[..., 0:1]
        xmin = boxes[..., 1:2]
        ymax = boxes[..., 2:3]
        xmax = boxes[..., 3:4]
        width = xmax - xmin
        height = ymax - ymin
        new_center_x = (xmin + xmax) / 2.0 + bbox_jitters[..., 0:1] * width
        new_center_y = (ymin + ymax) / 2.0 + bbox_jitters[..., 1:2] * height
        new_width = width * tf.exp(bbox_jitters[..., 2:3])
        new_height = height * tf.exp(bbox_jitters[..., 3:4])
        jittered_boxes = tf.concat([new_center_y - new_height * 0.5, 
            new_center_x - new_width * 0.5, new_center_y + new_height * 0.5,
            new_center_x + new_width * 0.5], axis=-1)
        return jittered_boxes
