def _adaptive_adams_step(self, vcabm_state, final_t):
    y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
    if next_t > final_t:
        next_t = final_t
    dt = next_t - prev_t[0]
    dt_cast = dt.to(y0[0])
    g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
    g = g.to(y0[0])
    p_next = tuple(y0_ + _scaled_dot_product(dt_cast, g[:max(1, order - 1)],
        phi_[:max(1, order - 1)]) for y0_, phi_ in zip(y0, tuple(zip(*phi))))
    next_f0 = self.func(next_t.to(p_next[0]), p_next, **self.f_options)
    implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)
    y_next = tuple(p_next_ + dt_cast * g[order - 1] * iphi_ for p_next_,
        iphi_ in zip(p_next, implicit_phi_p[order - 1]))
    tolerance = tuple(atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(
        y1_)) for atol_, rtol_, y0_, y1_ in zip(self.atol, self.rtol, y0,
        y_next))
    local_error = tuple(dt_cast * (g[order] - g[order - 1]) * iphi_ for
        iphi_ in implicit_phi_p[order])
    error_k = _compute_error_ratio(local_error, tolerance)
    accept_step = (torch.tensor(error_k) <= 1).all()
    if not accept_step:
        dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor,
            self.dfactor, order=order)
        return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next,
            prev_phi, order=order)
    next_f0 = self.func(next_t.to(p_next[0]), y_next, **self.f_options)
    implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)
    next_order = order
    if len(prev_t) <= 4 or order < 3:
        next_order = min(order + 1, 3, self.max_order)
    else:
        error_km1 = _compute_error_ratio(tuple(dt_cast * (g[order - 1] - g[
            order - 2]) * iphi_ for iphi_ in implicit_phi_p[order - 1]),
            tolerance)
        error_km2 = _compute_error_ratio(tuple(dt_cast * (g[order - 2] - g[
            order - 3]) * iphi_ for iphi_ in implicit_phi_p[order - 2]),
            tolerance)
        if min(error_km1 + error_km2) < max(error_k):
            next_order = order - 1
        elif order < self.max_order:
            error_kp1 = _compute_error_ratio(tuple(dt_cast * gamma_star[
                order] * iphi_ for iphi_ in implicit_phi_p[order]), tolerance)
            if max(error_kp1) < max(error_k):
                next_order = order + 1
    dt_next = dt if next_order > order else _optimal_step_size(dt, error_k,
        self.safety, self.ifactor, self.dfactor, order=order + 1)
    prev_f.appendleft(next_f0)
    prev_t.appendleft(next_t)
    return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next,
        implicit_phi, order=next_order)
