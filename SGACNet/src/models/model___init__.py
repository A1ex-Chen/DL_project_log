def __init__(self, mode, channels=None):
    super(Upsample, self).__init__()
    self.interp = nn.functional.interpolate
    if mode == 'bilinear':
        self.align_corners = False
    else:
        self.align_corners = None
    if 'learned-3x3' in mode:
        if mode == 'learned-3x3':
            self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
            self.conv = nn.Conv2d(channels, channels, groups=channels,
                kernel_size=3, padding=0)
        elif mode == 'learned-3x3-zeropad':
            self.pad = nn.Identity()
            self.conv = nn.Conv2d(channels, channels, groups=channels,
                kernel_size=3, padding=1)
        w = torch.tensor([[[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]]]])
        self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))
        with torch.no_grad():
            self.conv.bias.zero_()
        self.mode = 'nearest'
    else:
        self.pad = nn.Identity()
        self.conv = nn.Identity()
        self.mode = mode
