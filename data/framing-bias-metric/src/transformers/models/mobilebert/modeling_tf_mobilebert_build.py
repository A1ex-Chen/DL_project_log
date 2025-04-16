def build(self, input_shape):
    self.bias = self.add_weight(shape=(self.vocab_size,), initializer=
        'zeros', trainable=True, name='bias')
    self.dense = self.add_weight(shape=(self.config.hidden_size - self.
        config.embedding_size, self.vocab_size), initializer='zeros',
        trainable=True, name='dense/weight')
    self.decoder = self.add_weight(shape=(self.config.vocab_size, self.
        config.embedding_size), initializer='zeros', trainable=True, name=
        'decoder/weight')
    super().build(input_shape)
