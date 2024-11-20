def _initialize_aux_biases(self, cf=None):
    m = self.model[-1]
    for mi, mi2, s in zip(m.m, m.m2, m.stride):
        b = mi.bias.view(m.na, -1)
        b.data[:, 4] += math.log(8 / (640 / s) ** 2)
        b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)
            ) if cf is None else torch.log(cf / cf.sum())
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        b2 = mi2.bias.view(m.na, -1)
        b2.data[:, 4] += math.log(8 / (640 / s) ** 2)
        b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)
            ) if cf is None else torch.log(cf / cf.sum())
        mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)
