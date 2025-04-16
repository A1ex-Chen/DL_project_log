def _make_layers(self, in_planes):
    layers = []
    for x in self.cfg:
        out_planes = x if isinstance(x, int) else x[0]
        stride = 1 if isinstance(x, int) else x[1]
        layers.append(Block(in_planes, out_planes, stride))
        in_planes = out_planes
    return nn.Sequential(*layers)
