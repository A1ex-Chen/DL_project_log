def stem(self, x):
    for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.
        conv3, self.bn3)]:
        x = self.relu(bn(conv(x)))
    x = self.avgpool(x)
    return x
