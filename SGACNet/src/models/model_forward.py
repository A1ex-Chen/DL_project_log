def forward(self, x):
    size = int(x.shape[2] * 2), int(x.shape[3] * 2)
    x = self.interp(x, size, mode=self.mode, align_corners=self.align_corners)
    x = self.pad(x)
    x = self.conv(x)
    return x
