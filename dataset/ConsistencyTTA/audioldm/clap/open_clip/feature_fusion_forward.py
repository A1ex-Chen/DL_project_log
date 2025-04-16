def forward(self, x, residual):
    flag = False
    xa = x + residual
    if xa.size(0) == 1:
        xa = torch.cat([xa, xa], dim=0)
        flag = True
    xl = self.local_att(xa)
    xg = self.global_att(xa)
    xlg = xl + xg
    wei = self.sigmoid(xlg)
    xo = 2 * x * wei + 2 * residual * (1 - wei)
    if flag:
        xo = xo[0].unsqueeze(0)
    return xo
