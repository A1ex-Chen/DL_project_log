def forward_first_conv(self, x):
    x = ConvBNReLU(3, 64, kernel_size=7, stride=1, groups=1)
    return x
