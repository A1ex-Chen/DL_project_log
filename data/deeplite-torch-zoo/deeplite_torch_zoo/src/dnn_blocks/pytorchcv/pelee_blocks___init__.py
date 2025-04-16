def __init__(self, c1, c2, expansion_factor=2, bottleneck_size=1, k1=3, k2=
    3, act='relu'):
    super().__init__()
    inc_channels = (int(expansion_factor * c2) - c1) // 2
    mid_channels = int(inc_channels * bottleneck_size)
    self.branch1 = PeleeBranch1(in_channels=c1, out_channels=inc_channels,
        mid_channels=mid_channels, kernel_size=k1, act=act)
    self.branch2 = PeleeBranch2(in_channels=c1, out_channels=inc_channels,
        mid_channels=mid_channels, kernel_size=k2, act=act)
    self.conv = ConvBnAct(c1=int(expansion_factor * c2), c2=c2, k=1, act=act)
