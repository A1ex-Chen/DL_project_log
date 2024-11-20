def forward(self, x):
    return self.conv(x) + self.shortcut(x)
