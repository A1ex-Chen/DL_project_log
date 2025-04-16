def forward_once(self, x, profile=False):
    y, dt = [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j
                ]) for j in m.f]
        if not hasattr(self, 'traced'):
            self.traced = False
        if self.traced:
            if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m,
                IAuxDetect) or isinstance(m, IKeypoint):
                break
        if profile:
            c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
            o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[
                0] / 1000000000.0 * 2 if thop else 0
            for _ in range(10):
                m(x.copy() if c else x)
            t = time_synchronized()
            for _ in range(10):
                m(x.copy() if c else x)
            dt.append((time_synchronized() - t) * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
        x = m(x)
        y.append(x if m.i in self.save else None)
    if profile:
        print('%.1fms total' % sum(dt))
    return x
