def build(self, input_shape):
    if self.use_scale:
        self.scale = self.add_weight(name='scale', shape=(), initializer=
            init_ops.constant_initializer(self.use_scale), dtype=self.dtype,
            trainable=True)
    else:
        self.scale = None
    super(Attention, self).build(input_shape)
