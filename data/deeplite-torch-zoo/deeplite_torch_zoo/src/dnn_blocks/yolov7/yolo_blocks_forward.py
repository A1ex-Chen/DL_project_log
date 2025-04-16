def forward(self, x):
    x = self.conv_deconv(self.conv_strided(x))
    if self.gamma is not None:
        x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    return x
