def _make_layer(self, out_channels, num_blocks):
    layers = [DownBlock(self.in_channels, out_channels)]
    for i in range(num_blocks):
        layers.append(BasicBlock(out_channels))
        self.in_channels = out_channels
    return nn.Sequential(*layers)
