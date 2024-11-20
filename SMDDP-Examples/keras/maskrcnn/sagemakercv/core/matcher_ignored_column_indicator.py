def ignored_column_indicator(self):
    """Returns boolean column indicator where True means the colum is ignored.

    Returns:
      column_indicator: boolean vector which is True for all ignored column
      indices.
    """
    return tf.equal(self._match_results, -2)
