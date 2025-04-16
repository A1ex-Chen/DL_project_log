def __init__(self, c1, c2, k=3, s=1, shrink_factor=0.5):
    super(GhostBottleneck, self).__init__()
    c_ = c2 // 2
    self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1, shrink_factor=
        shrink_factor), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.
        Identity(), GhostConv(c_, c2, 1, 1, act=False, shrink_factor=
        shrink_factor))
    self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
        ConvBnAct(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
