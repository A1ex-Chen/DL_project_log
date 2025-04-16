def _linear_interp(self, t0, t1, y0, y1, t):
    if t == t0:
        return y0
    if t == t1:
        return y1
    t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
    slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_ in zip(y0, y1))
    return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
