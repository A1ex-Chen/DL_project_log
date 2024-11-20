def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=0.5):
    super().__init__()
    mid_channels = int(out_channels * expand_ratio)
    self.conv_1 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
    self.conv_2 = ConvBNHS(in_channels, mid_channels, 1, 1, 0)
    self.conv_3 = ConvBNHS(2 * mid_channels, out_channels, 1, 1, 0)
    self.blocks = DarknetBlock(mid_channels, mid_channels, kernel_size, 1.0)
