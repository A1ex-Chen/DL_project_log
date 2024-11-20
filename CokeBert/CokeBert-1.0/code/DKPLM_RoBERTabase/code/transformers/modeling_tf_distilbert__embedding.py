def _embedding(self, inputs, inputs_embeds=None, training=False):
    """
        Parameters
        ----------
        input_ids: tf.Tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: tf.Tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
    if not isinstance(inputs, (tuple, list)):
        input_ids = inputs
        position_ids = None
    else:
        input_ids, position_ids = inputs
    if input_ids is not None:
        seq_length = tf.shape(input_ids)[1]
    else:
        seq_length = tf.shape(inputs_embeds)[1]
    if position_ids is None:
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
    if inputs_embeds is None:
        inputs_embeds = tf.gather(self.word_embeddings, input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    embeddings = inputs_embeds + position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings, training=training)
    return embeddings
