def __init__(self, nc=80, ch=(), act='relu'):
    super().__init__()
    self.nc = nc
    self.nl = len(ch)
    self.reg_max = 16
    self.no = nc + self.reg_max * 4
    self.stride = torch.zeros(self.nl)
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
    self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3, act=act), Conv(c2,
        c2, 3, act=act), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
    self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3, act=act), Conv(c3,
        c3, 3, act=act), nn.Conv2d(c3, self.nc, 1)) for x in ch)
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
