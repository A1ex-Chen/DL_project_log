def reset(self, c, stride):
    self.layers = nn.ModuleList()
    for primitive in LINEAR_PRIMITIVES:
        layer = OPS[primitive](c, stride, False)
        if 'pool' in primitive:
            layer = nn.Sequential(layer, nn.BatchNorm1d(c, affine=False))
        self.layers.append(layer)
