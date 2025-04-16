def __init__(self, c1, c2, k=1, s=1, g=1, dw_k=5, dw_s=1, act='relu',
    shrink_factor=0.5, residual=False, dfc=False):
    super(GhostConv, self).__init__()
    c_ = int(c2 * shrink_factor)
    self.residual = residual
    self.dfc = None
    if dfc:
        self.dfc = DFCModule(c1, c2, k, s)
    self.single_conv = False
    if c_ < 2:
        self.single_conv = True
        self.cv1 = ConvBnAct(c1, c2, k, s, p=None, g=g, act=act)
    else:
        self.cv1 = ConvBnAct(c1, c_, k, s, p=None, g=g, act=act)
        self.cv2 = ConvBnAct(c_, c_, dw_k, dw_s, p=None, g=c_, act=act)
