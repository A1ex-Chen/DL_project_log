def _make_layers(self, in_planes):
    layers = []
    for expansion, out_planes, num_blocks, stride in self.cfg:
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(Block(in_planes, out_planes, expansion, stride))
            in_planes = out_planes
    return nn.Sequential(*layers)
