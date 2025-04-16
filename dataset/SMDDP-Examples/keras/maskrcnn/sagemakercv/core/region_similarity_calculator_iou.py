def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0
        ) - intersections
    return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(
        intersections), tf.truediv(intersections, unions))
