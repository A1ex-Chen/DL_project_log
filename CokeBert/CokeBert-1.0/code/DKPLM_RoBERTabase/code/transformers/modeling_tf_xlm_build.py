def build(self, input_shape):
    self.bias = self.add_weight(shape=(self.n_words,), initializer='zeros',
        trainable=True, name='bias')
    super(TFXLMPredLayer, self).build(input_shape)
