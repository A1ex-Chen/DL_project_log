def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
    super(ST2CSPC, self).__init__()
    c_ = int(c2 * e)
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c1, c_, 1, 1)
    self.cv3 = Conv(c_, c_, 1, 1)
    self.cv4 = Conv(2 * c_, c2, 1, 1)
    num_heads = c_ // 32
    self.m = SwinTransformer2Block(c_, c_, num_heads, n)
