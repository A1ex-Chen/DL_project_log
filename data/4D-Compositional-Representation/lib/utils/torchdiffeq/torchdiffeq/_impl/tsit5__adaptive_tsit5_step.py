def _adaptive_tsit5_step(self, rk_state):
    """Take an adaptive Runge-Kutta step to integrate the ODE."""
    y0, f0, _, t0, dt, _ = rk_state
    assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
    for y0_ in y0:
        assert _is_finite(torch.abs(y0_)
            ), 'non-finite values in state `y`: {}'.format(y0_)
    y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt,
        tableau=_TSITOURAS_TABLEAU)
    error_tol = tuple(self.atol + self.rtol * torch.max(torch.abs(y0_),
        torch.abs(y1_)) for y0_, y1_ in zip(y0, y1))
    tensor_error_ratio = tuple(y1_error_ / error_tol_ for y1_error_,
        error_tol_ in zip(y1_error, error_tol))
    sq_error_ratio = tuple(torch.mul(tensor_error_ratio_,
        tensor_error_ratio_) for tensor_error_ratio_ in tensor_error_ratio)
    mean_error_ratio = sum(torch.sum(sq_error_ratio_) for sq_error_ratio_ in
        sq_error_ratio) / sum(sq_error_ratio_.numel() for sq_error_ratio_ in
        sq_error_ratio)
    accept_step = mean_error_ratio <= 1
    y_next = y1 if accept_step else y0
    f_next = f1 if accept_step else f0
    t_next = t0 + dt if accept_step else t0
    dt_next = _optimal_step_size(dt, mean_error_ratio, self.safety, self.
        ifactor, self.dfactor)
    k_next = k if accept_step else self.rk_state.interp_coeff
    rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, k_next)
    return rk_state
