def _forward_once(self, x, profile=False, visualize=False):
    y, dt = [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j
                ]) for j in m.f]
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)
        y.append(x if m.i in self.save else None)
    return x
