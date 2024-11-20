def forward(self, x):
    y = self.pre_bn(x)
    y = self.pw1_act(self.pw1(y))
    y = self.dw_act(self.dw(y))
    y = self.pw2(y)
    x = x + self.drop_path(y)
    y = self.premlp_bn(x)
    y = self.mlp(y)
    x = x + self.drop_path(y)
    return x
