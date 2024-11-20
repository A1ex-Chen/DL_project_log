def build(self, input_shape):
    self.bias = self.add_weight(shape=(self.vocab_size,), initializer=
        'zeros', trainable=True, name='bias')
    self.decoder_bias = self.add_weight(shape=(self.vocab_size,),
        initializer='zeros', trainable=True, name='decoder/bias')
    super().build(input_shape)
