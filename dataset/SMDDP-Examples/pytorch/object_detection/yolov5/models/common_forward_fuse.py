def forward_fuse(self, x):
    return self.act(self.conv(x))
