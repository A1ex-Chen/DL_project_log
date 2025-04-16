def set_layer_lr_decay(self, layer_lr_decay):
    """Sets the learning rate decay for each layer in the TinyViT model."""
    decay_rate = layer_lr_decay
    depth = sum(self.depths)
    lr_scales = [(decay_rate ** (depth - i - 1)) for i in range(depth)]

    def _set_lr_scale(m, scale):
        """Sets the learning rate scale for each layer in the model based on the layer's depth."""
        for p in m.parameters():
            p.lr_scale = scale
    self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
    i = 0
    for layer in self.layers:
        for block in layer.blocks:
            block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
            i += 1
        if layer.downsample is not None:
            layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1])
                )
    assert i == depth
    for m in [self.norm_head, self.head]:
        m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
    for k, p in self.named_parameters():
        p.param_name = k

    def _check_lr_scale(m):
        """Checks if the learning rate scale attribute is present in module's parameters."""
        for p in m.parameters():
            assert hasattr(p, 'lr_scale'), p.param_name
    self.apply(_check_lr_scale)
