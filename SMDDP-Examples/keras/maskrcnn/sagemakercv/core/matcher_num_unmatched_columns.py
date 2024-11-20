def num_unmatched_columns(self):
    """Returns number (int32 scalar tensor) of unmatched columns."""
    return tf.size(input=self.unmatched_column_indices())
