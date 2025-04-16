def matched_row_indices(self):
    """Returns row indices that match some column.

    The indices returned by this op are ordered so as to be in correspondence
    with the output of matched_column_indicator().  For example if
    self.matched_column_indicator() is [0,2], and self.matched_row_indices() is
    [7, 3], then we know that column 0 was matched to row 7 and column 2 was
    matched to row 3.

    Returns:
      row_indices: int32 tensor of shape [K] with row indices.
    """
    return self._reshape_and_cast(tf.gather(self._match_results, self.
        matched_column_indices()))
