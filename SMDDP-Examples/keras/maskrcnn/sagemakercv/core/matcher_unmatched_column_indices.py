def unmatched_column_indices(self):
    """Returns column indices that do not match any row.

    The indices returned by this op are always sorted in increasing order.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return self._reshape_and_cast(tf.where(tf.equal(self._match_results, -1)))
