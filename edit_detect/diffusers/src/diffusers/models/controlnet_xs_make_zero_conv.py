def make_zero_conv(in_channels, out_channels=None):
    return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))
