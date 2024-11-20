def _make_layer(self, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(Block(self.in_planes, self.cardinality, self.
            bottleneck_width, stride))
        self.in_planes = (Block.expansion * self.cardinality * self.
            bottleneck_width)
    self.bottleneck_width *= 2
    return nn.Sequential(*layers)
