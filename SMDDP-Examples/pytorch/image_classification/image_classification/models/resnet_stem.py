def stem(self, x):
    x = self.conv1(x)
    if self.bn1 is not None:
        x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    return x
