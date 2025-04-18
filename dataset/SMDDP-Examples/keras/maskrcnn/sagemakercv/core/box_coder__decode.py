@abstractmethod
def _decode(self, rel_codes, anchors):
    """Method to be overriden by implementations.

    Args:
      rel_codes: a tensor representing N relative-encoded boxes
      anchors: BoxList of anchors

    Returns:
      boxlist: BoxList holding N boxes encoded in the ordinary way (i.e.,
        with corners y_min, x_min, y_max, x_max)
    """
    pass
