def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    if self.drop:
        x1 = self.dropblock(self.layer1(x))
        x2 = self.dropblock(self.layer2(x1))
    else:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    return x2, x3, x4
