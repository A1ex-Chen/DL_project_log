def create_position_ids_from_input_ids(self, x):
    """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: tf.Tensor

        Returns: tf.Tensor
        """
    mask = tf.cast(tf.math.not_equal(x, self.padding_idx), dtype=tf.int32)
    incremental_indices = tf.math.cumsum(mask, axis=1) * mask
    return incremental_indices + self.padding_idx
