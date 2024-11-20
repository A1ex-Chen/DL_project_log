def forward(self, x):
    if self.conv is not None:
        x = self.conv(x)
    b, _, w, h = x.shape
    p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
    return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b,
        self.c2, w, h)
