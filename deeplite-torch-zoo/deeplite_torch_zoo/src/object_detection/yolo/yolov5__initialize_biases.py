def _initialize_biases(self, cf=None):
    m = self.model[-1]
    for mi, s in zip(m.m, m.stride):
        b = mi.bias.view(m.na, -1)
        b.data[:, 4] += math.log(8 / (640 / s) ** 2)
        b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)
            ) if cf is None else torch.log(cf / cf.sum())
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
