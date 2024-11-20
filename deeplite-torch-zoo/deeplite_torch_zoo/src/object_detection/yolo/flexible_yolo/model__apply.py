def _apply(self, fn):
    self = nn.Module._apply(self, fn)
    m = self.detection
    if isinstance(m, Detect):
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
    return self
