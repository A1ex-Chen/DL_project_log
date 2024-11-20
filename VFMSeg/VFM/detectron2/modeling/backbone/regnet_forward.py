def forward(self, x):
    """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
    assert x.dim(
        ) == 4, f'Model takes an input of shape (N, C, H, W). Got {x.shape} instead!'
    outputs = {}
    x = self.stem(x)
    if 'stem' in self._out_features:
        outputs['stem'] = x
    for stage, name in self.stages_and_names:
        x = stage(x)
        if name in self._out_features:
            outputs[name] = x
    return outputs
