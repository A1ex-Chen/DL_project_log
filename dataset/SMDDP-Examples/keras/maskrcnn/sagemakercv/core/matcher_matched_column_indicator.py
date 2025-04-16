def matched_column_indicator(self):
    """Returns column indices that are matched.

    Returns:
      column_indices: int32 tensor of shape [K] with column indices.
    """
    return tf.greater_equal(self._match_results, 0)
