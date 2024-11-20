def forward_first_conv(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act(x)
    return x
