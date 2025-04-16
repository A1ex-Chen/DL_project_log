def unpad(self, x):
    ht, wd = x.shape[-2:]
    c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
    return x[..., c[0]:c[1], c[2]:c[3]]
