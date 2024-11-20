def forward(self, x):
    c1 = self.C1(x)
    c2 = self.C2(c1)
    conv1 = self.conv1(c2)
    c3 = self.C3(conv1)
    conv2 = self.conv2(c3)
    c4 = self.C4(conv2)
    conv3 = self.conv3(c4)
    c5 = self.C5(conv3)
    conv4 = self.conv4(c5)
    sppf = self.sppf(conv4)
    return c3, c4, sppf
