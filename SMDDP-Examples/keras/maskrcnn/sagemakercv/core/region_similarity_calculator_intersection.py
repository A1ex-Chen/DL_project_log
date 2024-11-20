def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
    y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxlist1.get(),
        num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxlist2.get(),
        num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(a=y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(a=y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin
        )
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(a=x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(a=x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths
