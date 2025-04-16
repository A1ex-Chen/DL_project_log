def forward(self, x):
    if self.stride == 1:
        return x.mul(0.0)
    return x[:, :, ::self.stride].mul(0.0)
