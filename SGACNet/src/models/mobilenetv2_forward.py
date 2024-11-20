def forward(self, x):
    x = self.features(x)
    x = x.mean([2, 3])
    x = self.classifier(x)
    x_layer1 = self.forward_resblock(x, self.layer1)
    x_layer2 = self.forward_resblock(x_layer1, self.layer2)
    x_layer3 = self.forward_resblock(x_layer2, self.layer3)
    x_layer4 = self.forward_resblock(x_layer3, self.layer4)
    self.skip3_channels = x_layer3.size()[1]
    self.skip2_channels = x_layer2.size()[1]
    self.skip1_channels = x_layer1.size()[1]
    return x
