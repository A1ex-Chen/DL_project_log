def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.
        LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias
        =False), nn.GroupNorm(min(32, ndf * 2), ndf * 2), nn.LeakyReLU(0.2,
        inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn
        .GroupNorm(min(32, ndf * 4), ndf * 4), nn.LeakyReLU(0.2, inplace=
        True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.
        GroupNorm(min(32, ndf * 8), ndf * 8), nn.LeakyReLU(0.2, inplace=
        True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())
