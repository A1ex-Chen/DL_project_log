def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel', shape=(input_shape[1],
        self.output_dim), initializer='uniform', trainable=True)
    super(Attention, self).build(input_shape)
