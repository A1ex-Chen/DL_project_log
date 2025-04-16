def init_weights(self):
    init_layer(self.conv1)
    init_bn(self.bn1)
    init_layer(self.conv2)
    init_bn(self.bn2)
    nn.init.constant_(self.bn2.weight, 0)
