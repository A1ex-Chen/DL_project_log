def build(self, input_shape):
    self.bias = self.add_weight(shape=(self.vocab_size,), initializer=
        'zeros', trainable=True, name='bias')
    super(TFRobertaLMHead, self).build(input_shape)
