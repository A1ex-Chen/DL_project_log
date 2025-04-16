def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
    super().__init__()
    self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=
        kernel_size // 2, groups=c)
    self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-06)
    self.channelwise = nn.Sequential(nn.Linear(c + c_skip, c * 4), nn.GELU(
        ), GlobalResponseNorm(c * 4), nn.Dropout(dropout), nn.Linear(c * 4, c))
