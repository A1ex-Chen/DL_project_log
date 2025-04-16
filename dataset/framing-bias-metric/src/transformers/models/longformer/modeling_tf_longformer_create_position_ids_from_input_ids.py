def create_position_ids_from_input_ids(self, input_ids):
    """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor

        Returns: tf.Tensor
        """
    input_ids_shape = shape_list(input_ids)
    if len(input_ids_shape) == 3:
        input_ids = tf.reshape(input_ids, (input_ids_shape[0] *
            input_ids_shape[1], input_ids_shape[2]))
    mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=tf
        .int32)
    incremental_indices = tf.math.cumsum(mask, axis=1) * mask
    return incremental_indices + self.padding_idx
