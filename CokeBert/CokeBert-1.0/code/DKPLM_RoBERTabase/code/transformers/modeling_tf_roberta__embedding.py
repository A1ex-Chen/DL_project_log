def _embedding(self, inputs, training=False):
    """Applies embedding based on inputs tensor."""
    input_ids, position_ids, token_type_ids, inputs_embeds = inputs
    if input_ids is not None:
        seq_length = tf.shape(input_ids)[1]
    else:
        seq_length = tf.shape(inputs_embeds)[1]
    if position_ids is None:
        position_ids = tf.range(self.padding_idx + 1, seq_length + self.
            padding_idx + 1, dtype=tf.int32)[tf.newaxis, :]
    return super(TFRobertaEmbeddings, self)._embedding([input_ids,
        position_ids, token_type_ids, inputs_embeds], training=training)
