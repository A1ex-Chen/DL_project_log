def _apply(self, fn):
    self = super()._apply(fn)
    if self.pt:
        m = self.model.model.model[-1] if self.dmb else self.model.model[-1]
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
    return self
