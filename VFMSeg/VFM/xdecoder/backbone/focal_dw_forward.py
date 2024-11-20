def forward(self, x):
    """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
    assert x.dim(
        ) == 4, f'SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!'
    outputs = {}
    y = super().forward(x)
    for k in y.keys():
        if k in self._out_features:
            outputs[k] = y[k]
    return outputs
