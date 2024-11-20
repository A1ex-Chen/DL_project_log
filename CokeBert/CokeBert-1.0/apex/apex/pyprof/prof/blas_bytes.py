def bytes(self):
    m, n, k = self.m, self.n, self.k
    return Utility.typeToBytes(self.type) * (m * n + m * k + n * k)
