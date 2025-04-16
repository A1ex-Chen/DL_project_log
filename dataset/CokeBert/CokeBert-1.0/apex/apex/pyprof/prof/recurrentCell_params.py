def params(self):
    if self.gemm is None:
        p = OrderedDict([('cell', self.cell), ('X', self.inp), ('H', self.
            hid), ('B', self.b), ('type', self.type)])
    else:
        assert self.m is not None
        assert self.n is not None
        assert self.k is not None
        p = OrderedDict([('gemm', self.gemm), ('M', self.m), ('N', self.n),
            ('K', self.k), ('type', self.type)])
    return p
