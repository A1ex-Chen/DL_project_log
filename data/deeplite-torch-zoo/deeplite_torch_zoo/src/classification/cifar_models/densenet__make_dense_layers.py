def _make_dense_layers(self, block, in_planes, nblock):
    layers = []
    for i in range(nblock):
        layers.append(block(in_planes, self.growth_rate))
        in_planes += self.growth_rate
    return nn.Sequential(*layers)
