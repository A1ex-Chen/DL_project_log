def ignored_column_indices(self):
    """Returns column indices that are ignored (neither Matched nor Unmatched).

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(self.ignored_column_indicator()))
