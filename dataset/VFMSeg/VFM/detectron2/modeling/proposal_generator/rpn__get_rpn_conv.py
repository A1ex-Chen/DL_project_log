def _get_rpn_conv(self, in_channels, out_channels):
    return Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, activation=nn.ReLU())
