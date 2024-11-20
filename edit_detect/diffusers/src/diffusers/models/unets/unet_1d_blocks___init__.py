def __init__(self, in_channels: int, out_channels: int, mid_channels:
    Optional[int]=None):
    super().__init__()
    mid_channels = in_channels if mid_channels is None else mid_channels
    resnets = [ResConvBlock(2 * in_channels, mid_channels, mid_channels),
        ResConvBlock(mid_channels, mid_channels, mid_channels),
        ResConvBlock(mid_channels, mid_channels, out_channels, is_last=True)]
    self.resnets = nn.ModuleList(resnets)