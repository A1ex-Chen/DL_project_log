def forward(self, x):
    """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
    assert x.dim(
        ) == 4, f'ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!'
    outputs = {}
    x = self.stem(x)
    if 'stem' in self._out_features:
        outputs['stem'] = x
    for name, stage in zip(self.stage_names, self.stages):
        x = stage(x)
        if name in self._out_features:
            outputs[name] = x
    if self.num_classes is not None:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        if 'linear' in self._out_features:
            outputs['linear'] = x
    return outputs
