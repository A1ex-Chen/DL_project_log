def num_ignored_columns(self):
    """Returns number (int32 scalar tensor) of matched columns."""
    return tf.size(input=self.ignored_column_indices())
