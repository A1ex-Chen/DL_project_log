def forward(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x_down2 = self.act(x)
    x = self.maxpool(x_down2)
    x_layer1 = self.forward_resblock(x, self.layer1)
    x_layer2 = self.forward_resblock(x_layer1, self.layer2)
    x_layer3 = self.forward_resblock(x_layer2, self.layer3)
    x_layer4 = self.forward_resblock(x_layer3, self.layer4)
    if self.replace_stride_with_dilation == [False, False, False]:
        features = [x_layer4, x_layer3, x_layer2, x_layer1]
        self.skip3_channels = x_layer3.size()[1]
        self.skip2_channels = x_layer2.size()[1]
        self.skip1_channels = x_layer1.size()[1]
    elif self.replace_stride_with_dilation == [False, True, True]:
        features = [x, x_layer1, x_down2]
        self.skip3_channels = x_layer3.size()[1]
        self.skip2_channels = x_layer2.size()[1]
        self.skip1_channels = x_layer1.size()[1]
    return features
