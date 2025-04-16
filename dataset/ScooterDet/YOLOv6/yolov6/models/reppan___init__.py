def __init__(self, in_channels, unified_channels):
    super().__init__()
    self.reduce_layer0 = ConvBNHS(in_channels=in_channels[0], out_channels=
        unified_channels, kernel_size=1, stride=1, padding=0)
    self.reduce_layer1 = ConvBNHS(in_channels=in_channels[1], out_channels=
        unified_channels, kernel_size=1, stride=1, padding=0)
    self.reduce_layer2 = ConvBNHS(in_channels=in_channels[2], out_channels=
        unified_channels, kernel_size=1, stride=1, padding=0)
    self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
    self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
    self.Csp_p4 = CSPBlock(in_channels=unified_channels * 2, out_channels=
        unified_channels, kernel_size=5)
    self.Csp_p3 = CSPBlock(in_channels=unified_channels * 2, out_channels=
        unified_channels, kernel_size=5)
    self.Csp_n3 = CSPBlock(in_channels=unified_channels * 2, out_channels=
        unified_channels, kernel_size=5)
    self.Csp_n4 = CSPBlock(in_channels=unified_channels * 2, out_channels=
        unified_channels, kernel_size=5)
    self.downsample2 = DPBlock(in_channel=unified_channels, out_channel=
        unified_channels, kernel_size=5, stride=2)
    self.downsample1 = DPBlock(in_channel=unified_channels, out_channel=
        unified_channels, kernel_size=5, stride=2)
    self.p6_conv_1 = DPBlock(in_channel=unified_channels, out_channel=
        unified_channels, kernel_size=5, stride=2)
    self.p6_conv_2 = DPBlock(in_channel=unified_channels, out_channel=
        unified_channels, kernel_size=5, stride=2)
