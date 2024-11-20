def bytesFlops(self):
    m = self.m
    n = Utility.numElems(self.n)
    k = self.k
    if self.op_ == 'linear':
        if self.dir == 'fprop':
            f = m * n * k * 2
            b = m * n + m * k + n * k * Utility.typeToBytes(self.type)
        elif self.dir == 'bprop':
            if self.sub == 0:
                f = m * n * k * 2
                b = m * n + m * k + n * k * Utility.typeToBytes(self.type)
            elif self.sub == 1:
                f = m * n * k * 2
                b = m * n + m * k + n * k * Utility.typeToBytes(self.type)
            else:
                f = 0
                b = 0
        else:
            assert False
    elif self.op_ == 'bias':
        f = m * n
        b = 2 * m * n * Utility.typeToBytes(self.type)
    else:
        assert False
    return b, f
