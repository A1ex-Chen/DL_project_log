def build(self, input_shape):
    """Build shared word embedding layer """
    with tf.name_scope('word_embeddings'):
        self.word_embeddings = self.add_weight('weight', shape=[self.
            vocab_size, self.hidden_size], initializer=get_initializer(self
            .initializer_range))
    super().build(input_shape)
