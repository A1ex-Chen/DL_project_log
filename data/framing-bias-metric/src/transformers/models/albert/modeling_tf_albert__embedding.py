def _embedding(self, input_ids, position_ids, token_type_ids, inputs_embeds,
    training=False):
    """Applies embedding based on inputs tensor."""
    assert not (input_ids is None and inputs_embeds is None)
    if input_ids is not None:
        input_shape = shape_list(input_ids)
    else:
        input_shape = shape_list(inputs_embeds)[:-1]
    seq_length = input_shape[1]
    if position_ids is None:
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
    if token_type_ids is None:
        token_type_ids = tf.fill(input_shape, 0)
    if inputs_embeds is None:
        inputs_embeds = tf.gather(self.word_embeddings, input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings, training=training)
    return embeddings
