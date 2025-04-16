def _embedding(self, input_ids, inputs_embeds, training=False):
    """Applies embedding based on inputs tensor."""
    assert not (input_ids is None and inputs_embeds is None)
    if inputs_embeds is None:
        inputs_embeds = tf.gather(self.word_embeddings, input_ids)
    embeddings = self.layer_norm(inputs_embeds)
    embeddings = self.dropout(embeddings, training=training)
    return embeddings
