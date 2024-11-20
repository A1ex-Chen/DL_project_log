def encode(self, boxes, anchors):
    """Encode a box list relative to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded
      anchors: BoxList of N anchors

    Returns:
      a tensor representing N relative-encoded boxes
    """
    return self._encode(boxes, anchors)
