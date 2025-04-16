def build(self, input_shape):
    """Build shared word embedding layer """
    self.weight = self.add_weight('weight', shape=(input_shape[-1],),
        initializer='ones')
    super().build(input_shape)
