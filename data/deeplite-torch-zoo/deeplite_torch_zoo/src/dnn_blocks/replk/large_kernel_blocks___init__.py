def __init__(self, c1, c2, k=31, small_kernel=5, dw_ratio=1.0, mlp_ratio=
    4.0, drop_path=0.0, activation=nn.ReLU):
    super().__init__()
    self.pre_bn = nn.BatchNorm2d(c1)
    self.pw1 = ConvBnAct(c1, int(c1 * dw_ratio), 1, 1, 0, act=None)
    self.pw1_act = activation()
    self.dw = LargeKernelReparam(int(c1 * dw_ratio), k, small_kernel=
        small_kernel)
    self.dw_act = activation()
    self.pw2 = ConvBnAct(int(c1 * dw_ratio), c1, 1, 1, 0, act=None)
    self.premlp_bn = nn.BatchNorm2d(c1)
    self.mlp = MLP(in_channels=c1, hidden_channels=int(c1 * mlp_ratio))
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
