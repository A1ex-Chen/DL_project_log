def transpose_coordinates(self, scope=None):
    """Transpose the coordinate representation in a boxlist.

    Args:
      scope: name scope of the function.
    """
    y_min, x_min, y_max, x_max = tf.split(value=self.get(),
        num_or_size_splits=4, axis=1)
    self.set(tf.concat([x_min, y_min, x_max, y_max], 1))
