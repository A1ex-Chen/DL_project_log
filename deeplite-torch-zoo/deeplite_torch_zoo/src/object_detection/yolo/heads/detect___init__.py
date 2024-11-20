def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
    super().__init__()
    self.nc = nc
    self.no = nc + 5
    self.nl = len(anchors)
    self.na = len(anchors[0]) // 2
    self.grid = [torch.empty(0) for _ in range(self.nl)]
    self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
    self.register_buffer('anchors', torch.tensor(anchors).float().view(self
        .nl, -1, 2))
    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
    self.inplace = inplace
