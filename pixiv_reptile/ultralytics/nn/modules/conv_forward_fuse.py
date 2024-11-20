def forward_fuse(self, x):
    """Forward process."""
    return self.act(self.conv(x))
