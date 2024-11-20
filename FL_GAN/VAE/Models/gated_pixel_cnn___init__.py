def __init__(self, in_channels, channels, out_channels):
    super().__init__()
    self.conv_vstack = VerticalStackConv('A', in_channels, channels, 3,
        padding=1)
    self.conv_hstack = HorizontalStackConv('A', in_channels, channels, 3,
        padding=1)
    self.conv_layers = nn.ModuleList([GatedMaskedConv(channels),
        GatedMaskedConv(channels, dilation=2), GatedMaskedConv(channels),
        GatedMaskedConv(channels, dilation=4), GatedMaskedConv(channels),
        GatedMaskedConv(channels, dilation=2), GatedMaskedConv(channels)])
    self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
