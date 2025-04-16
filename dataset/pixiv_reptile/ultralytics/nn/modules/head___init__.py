def __init__(self, nc=80, ch=()):
    """Initializes the v10Detect object with the specified number of classes and input channels."""
    super().__init__(nc, ch)
    c3 = max(ch[0], min(self.nc, 100))
    self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x),
        Conv(x, c3, 1)), nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 
        1)), nn.Conv2d(c3, self.nc, 1)) for x in ch)
    self.one2one_cv3 = copy.deepcopy(self.cv3)
