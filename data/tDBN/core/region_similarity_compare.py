def compare(self, boxes1, boxes2):
    """Computes matrix of pairwise similarity between BoxLists.

    This op (to be overriden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.

    Note that this method simply measures similarity and does not explicitly
    perform a matching.

    Args:
      boxes1: [N, 5] [x,y,w,l,r] tensor.
      boxes2: [M, 5] [x,y,w,l,r] tensor.

    Returns:
      a (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    return self._compare(boxes1, boxes2)
