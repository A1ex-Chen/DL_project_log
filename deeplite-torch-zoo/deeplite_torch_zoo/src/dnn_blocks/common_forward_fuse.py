def forward_fuse(self, x):
    inp = x
    out = self.act(self.conv(x))
    if self.residual:
        if self.resize_identity:
            out = out + self.identity_conv(inp)
        else:
            out = out + inp
    return out
