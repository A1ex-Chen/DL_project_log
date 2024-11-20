def reset(self, c, stride, primitives, ops):
    self.layers = nn.ModuleList()
    for primitive in primitives:
        layer = ops[primitive](c, stride, False)
        if 'pool' in primitive:
            layer = nn.Sequential(layer, nn.BatchNorm1d(c, affine=False))
        self.layers.append(layer)
