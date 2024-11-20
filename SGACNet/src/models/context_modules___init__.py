def __init__(self, in_dim, out_dim, bins=(1, 2, 4), align_corners=False,
    activation=nn.ReLU(inplace=True), upsampling_mode='bilinear'):
    reduction_dim = in_dim // len(bins)
    super(PPContextModule, self).__init__()
    self.features = []
    self.upsampling_mode = upsampling_mode
    for bin in bins:
        self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin),
            ConvBNReLU(in_dim, reduction_dim, kernel_size=1, activation=
            activation)))
    self.features = nn.ModuleList(self.features)
    self.final_conv = ConvBNReLU(reduction_dim, out_dim, bias=False,
        kernel_size=3, padding=1)
    self.align_corners = align_corners
