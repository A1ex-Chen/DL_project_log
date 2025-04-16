def __init__(self, in_channels, mid_channels, out_channels, num_repeat=[1, 
    3, 7, 3]):
    super().__init__()
    out_channels[0] = 24
    self.conv_0 = ConvBNHS(in_channels=in_channels, out_channels=
        out_channels[0], kernel_size=3, stride=2, padding=1)
    self.lite_effiblock_1 = self.build_block(num_repeat[0], out_channels[0],
        mid_channels[1], out_channels[1])
    self.lite_effiblock_2 = self.build_block(num_repeat[1], out_channels[1],
        mid_channels[2], out_channels[2])
    self.lite_effiblock_3 = self.build_block(num_repeat[2], out_channels[2],
        mid_channels[3], out_channels[3])
    self.lite_effiblock_4 = self.build_block(num_repeat[3], out_channels[3],
        mid_channels[4], out_channels[4])
