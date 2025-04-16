def build(self, input_shape):
    """Build shared word embedding layer
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
    self.weight = self.add_weight('weight', shape=[self.vocab_size, self.
        hidden_size], initializer=get_initializer(self.initializer_range))
    super(TFSharedEmbeddings, self).build(input_shape)
