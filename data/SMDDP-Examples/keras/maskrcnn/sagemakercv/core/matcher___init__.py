def __init__(self, match_results):
    """Constructs a Match object.

    Args:
      match_results: Integer tensor of shape [N] with (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.

    Raises:
      ValueError: if match_results does not have rank 1 or is not an
        integer int32 scalar tensor
    """
    if match_results.shape.ndims != 1:
        raise ValueError('match_results should have rank 1')
    if match_results.dtype != tf.int32:
        raise ValueError(
            'match_results should be an int32 or int64 scalar tensor')
    self._match_results = match_results
