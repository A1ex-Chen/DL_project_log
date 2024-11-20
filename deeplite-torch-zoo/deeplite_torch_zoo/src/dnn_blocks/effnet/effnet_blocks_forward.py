def forward(self, x):
    return self.conv(x) if not self.shortcut else x + self.conv(x)
