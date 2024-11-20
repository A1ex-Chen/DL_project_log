def create_position_ids_from_inputs_embeds(self, inputs_embeds):
    """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: tf.Tensor

        Returns: tf.Tensor
        """
    seq_length = shape_list(inputs_embeds)[1]
    position_ids = tf.range(self.padding_idx + 1, seq_length + self.
        padding_idx + 1, dtype=tf.int32)[tf.newaxis, :]
    return position_ids
