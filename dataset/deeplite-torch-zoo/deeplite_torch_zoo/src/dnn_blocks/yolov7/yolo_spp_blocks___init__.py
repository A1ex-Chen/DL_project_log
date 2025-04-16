def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13),
    act='hswish'):
    super().__init__()
    c_ = int(2 * c2 * e)
    self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
    self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
    self.cv3 = ConvBnAct(c_, c_, 3, 1, act=act)
    self.cv4 = ConvBnAct(c_, c_, 1, 1, act=act)
    self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x //
        2) for x in k])
    self.cv5 = ConvBnAct(4 * c_, c_, 1, 1, act=act)
    self.cv6 = ConvBnAct(c_, c_, 3, 1, act=act)
    self.cv7 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
