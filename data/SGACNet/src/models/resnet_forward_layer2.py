def forward_layer2(self, x):
    x = self.forward_resblock(x, self.layer2)
    self.skip2_channels = x.size()[1]
    return x
