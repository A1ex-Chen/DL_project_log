def box_list_scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(value=boxlist.get(),
        num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(tf.concat([y_min, x_min, y_max, x_max
        ], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)
