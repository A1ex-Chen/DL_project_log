def flops(self):
    if self.gemm is not None:
        m, n, k = self.m, self.n, self.k
        f = 2 * m * n * k
    elif self.elems != 0:
        f = 0
    else:
        f = 0
    return f
