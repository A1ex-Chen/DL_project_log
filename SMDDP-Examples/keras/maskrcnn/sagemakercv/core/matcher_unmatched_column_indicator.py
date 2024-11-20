def unmatched_column_indicator(self):
    """Returns column indices that are unmatched.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return tf.equal(self._match_results, -1)
