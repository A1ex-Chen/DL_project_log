def __init__(self, f_channels, zq_channels):
    super().__init__()
    self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32,
        eps=1e-06, affine=True)
    self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=
        1, padding=0)
    self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=
        1, padding=0)
