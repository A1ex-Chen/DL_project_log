def forward_layer1(self, x):
    x = self.forward_resblock(x, self.layer1)
    self.skip1_channels = x.size()[1]
    return x
