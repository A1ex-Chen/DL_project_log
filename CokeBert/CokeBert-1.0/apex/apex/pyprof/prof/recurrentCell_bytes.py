def bytes(self):
    if self.gemm is not None:
        m, n, k, t = self.m, self.n, self.k, self.type
        b = (m * k + k * n + m * n) * Utility.typeToBytes(t)
    elif self.elems != 0:
        b = self.elems * Utility.typeToBytes(self.type)
    else:
        b = 0
    return b
