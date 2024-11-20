def build(self, input_shape):
    if not self.untie_r:
        self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head),
            initializer='zeros', trainable=True, name='r_w_bias')
        self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head),
            initializer='zeros', trainable=True, name='r_r_bias')
    super(TFTransfoXLMainLayer, self).build(input_shape)
