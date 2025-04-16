def _apply(self, fn):
    self = super()._apply(fn)
    m = self.model[-1]
    if isinstance(m, Detect):
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
    return self
