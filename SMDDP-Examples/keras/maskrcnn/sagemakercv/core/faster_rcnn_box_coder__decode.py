def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ty, tx, th, tw = tf.unstack(tf.transpose(a=rel_codes))
    if self._scale_factors:
        ty /= self._scale_factors[0]
        tx /= self._scale_factors[1]
        th /= self._scale_factors[2]
        tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0
    return box_list.BoxList(tf.transpose(a=tf.stack([ymin, xmin, ymax, xmax])))
