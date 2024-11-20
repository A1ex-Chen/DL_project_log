def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON
    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.math.log(w / wa)
    th = tf.math.log(h / ha)
    if self._scale_factors:
        ty *= self._scale_factors[0]
        tx *= self._scale_factors[1]
        th *= self._scale_factors[2]
        tw *= self._scale_factors[3]
    return tf.transpose(a=tf.stack([ty, tx, th, tw]))
