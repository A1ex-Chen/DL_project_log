def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act='relu', deploy=False):
    super(RepConv, self).__init__()
    self.deploy = deploy
    self.groups = g
    self.in_channels = c1
    self.out_channels = c2
    assert k == 3
    assert autopad(k, p) == 1
    padding_11 = autopad(k, p) - k // 2
    self.act = get_activation(act)
    if deploy:
        self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g,
            bias=True)
    else:
        self.rbr_identity = nn.BatchNorm2d(num_features=c1
            ) if c2 == c1 and s == 1 else None
        self.rbr_dense = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p
            ), groups=g, bias=False), nn.BatchNorm2d(num_features=c2))
        self.rbr_1x1 = nn.Sequential(nn.Conv2d(c1, c2, 1, s, padding_11,
            groups=g, bias=False), nn.BatchNorm2d(num_features=c2))
