def build(self, input_shape):
    """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
    super().build(input_shape)
    weight: np.ndarray = self._init_weight(self.input_dim, self.output_dim)
    self.set_weights([weight])
