def __init__(self, in_channels, out_channels, in_feature='res5'):
    super().__init__()
    self.num_levels = 2
    self.in_feature = in_feature
    self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
    self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
    for module in [self.p6, self.p7]:
        weight_init.c2_xavier_fill(module)
