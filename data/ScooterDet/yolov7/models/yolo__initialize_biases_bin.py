def _initialize_biases_bin(self, cf=None):
    m = self.model[-1]
    bc = m.bin_count
    for mi, s in zip(m.m, m.stride):
        b = mi.bias.view(m.na, -1)
        old = b[:, (0, 1, 2, bc + 3)].data
        obj_idx = 2 * bc + 4
        b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
        b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)
        b[:, obj_idx + 1:].data += math.log(0.6 / (m.nc - 0.99)
            ) if cf is None else torch.log(cf / cf.sum())
        b[:, (0, 1, 2, bc + 3)].data = old
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
