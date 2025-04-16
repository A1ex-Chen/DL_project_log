def set_input_embeddings(self, new_embeddings):
    self.shared.weight = new_embeddings
    self.shared.vocab_size = self.shared.weight.shape[0]
    with tf.compat.v1.variable_scope('shared') as shared_abs_scope_name:
        pass
    embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=
        shared_abs_scope_name)
    self.encoder.set_embed_tokens(embed_tokens)
