def forward_layer4(self, x):
    x = self.forward_resblock(x, self.layer4)
    return x
