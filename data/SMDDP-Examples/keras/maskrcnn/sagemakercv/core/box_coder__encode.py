@abstractmethod
def _encode(self, boxes, anchors):
    """Method to be overriden by implementations.

    Args:
      boxes: BoxList holding N boxes to be encoded
      anchors: BoxList of N anchors

    Returns:
      a tensor representing N relative-encoded boxes
    """
    pass
