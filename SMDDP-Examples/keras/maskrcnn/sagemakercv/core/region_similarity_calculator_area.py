def area(boxlist, scope=None):
    """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
    y_min, x_min, y_max, x_max = tf.split(value=boxlist.get(),
        num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])
