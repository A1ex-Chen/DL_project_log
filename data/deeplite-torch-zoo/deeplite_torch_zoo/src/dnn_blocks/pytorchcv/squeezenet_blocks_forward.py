def forward(self, x):
    if self.resize_identity:
        identity = self.identity_conv(x)
    else:
        identity = x
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = x + identity
    x = self.activ(x)
    return x
