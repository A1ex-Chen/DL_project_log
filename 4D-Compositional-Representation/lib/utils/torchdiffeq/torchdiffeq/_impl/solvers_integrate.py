def integrate(self, t):
    _assert_increasing(t)
    t = t.type_as(self.y0[0])
    time_grid = self.grid_constructor(self.func, self.y0, t)
    assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
    time_grid = time_grid.to(self.y0[0])
    solution = [self.y0]
    j = 1
    y0 = self.y0
    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
        dy = self.step_func(self.func, t0, t1 - t0, y0, self.f_options)
        y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))
        y0 = y1
        while j < len(t) and t1 >= t[j]:
            solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
            j += 1
    return tuple(map(torch.stack, tuple(zip(*solution))))
