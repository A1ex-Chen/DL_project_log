def forward(self, x):
    ret = self[self._pos](x)
    self._pos = (self._pos + 1) % len(self)
    if self._affine:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        return ret * w + b
    else:
        return ret
