def forward_layer3(self, x):
    x = self.forward_resblock(x, self.layer3)
    self.skip3_channels = x.size()[1]
    return x
