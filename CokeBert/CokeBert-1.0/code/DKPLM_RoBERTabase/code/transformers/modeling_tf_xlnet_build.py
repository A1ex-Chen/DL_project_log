def build(self, input_shape):
    initializer = get_initializer(self.initializer_range)
    self.mask_emb = self.add_weight(shape=(1, 1, self.d_model), initializer
        =initializer, trainable=True, name='mask_emb')
