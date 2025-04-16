def __init__(self, c1, c2, transformer_block, n=1, act='relu', e=0.5):
    super(STCSPC, self).__init__()
    c_ = int(c2 * e)
    self.cv1 = ConvBnAct(c1, c_, 1, 1, act=act)
    self.cv2 = ConvBnAct(c1, c_, 1, 1, act=act)
    self.cv3 = ConvBnAct(c_, c_, 1, 1, act=act)
    self.cv4 = ConvBnAct(2 * c_, c2, 1, 1, act=act)
    num_heads = max(c_ // 32, 1)
    self.m = transformer_block(c_, c_, num_heads, n, act=act)
