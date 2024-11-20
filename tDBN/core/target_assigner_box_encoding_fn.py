def box_encoding_fn(boxes, anchors):
    return self._box_coder.encode(boxes, anchors)
