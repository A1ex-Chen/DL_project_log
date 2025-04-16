def _apply(self, fn):
    self = super()._apply(fn)
    self.detect.stride = fn(self.detect.stride)
    self.detect.grid = list(map(fn, self.detect.grid))
    return self
