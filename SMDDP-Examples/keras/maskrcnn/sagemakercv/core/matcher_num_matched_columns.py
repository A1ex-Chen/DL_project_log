def num_matched_columns(self):
    """Returns number (int32 scalar tensor) of matched columns."""
    return tf.size(input=self.matched_column_indices())
