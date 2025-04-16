def step_func(self, func, t, dt, y):
    self._update_history(t, func(t, y))
    order = min(len(self.prev_f), self.max_order - 1)
    if order < _MIN_ORDER - 1:
        dy = rk_common.rk4_alt_step_func(func, t, dt, y, k1=self.prev_f[0])
        return dy
    else:
        bashforth_coeffs = _BASHFORTH_COEFFICIENTS[order]
        ab_div = _DIVISOR[order]
        dy = tuple(dt * _scaled_dot_product(1 / ab_div, bashforth_coeffs,
            f_) for f_ in zip(*self.prev_f))
        if self.implicit:
            moulton_coeffs = _MOULTON_COEFFICIENTS[order + 1]
            am_div = _DIVISOR[order + 1]
            delta = tuple(dt * _scaled_dot_product(1 / am_div,
                moulton_coeffs[1:], f_) for f_ in zip(*self.prev_f))
            converged = False
            for _ in range(self.max_iters):
                dy_old = dy
                f = func(t + dt, tuple(y_ + dy_ for y_, dy_ in zip(y, dy)))
                dy = tuple(dt * (moulton_coeffs[0] / am_div) * f_ + delta_ for
                    f_, delta_ in zip(f, delta))
                converged = _has_converged(dy_old, dy, self.rtol, self.atol)
                if converged:
                    break
            if not converged:
                print(
                    'Warning: Functional iteration did not converge. Solution may be incorrect.'
                    , file=sys.stderr)
                self.prev_f.pop()
            self._update_history(t, f)
        return dy
