def _adaptive_dopri5_step(self, rk_state):
    """Take an adaptive Runge-Kutta step to integrate the ODE."""
    y0, f0, _, t0, dt, interp_coeff = rk_state
    assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
    for y0_ in y0:
        assert _is_finite(torch.abs(y0_)
            ), 'non-finite values in state `y`: {}'.format(y0_)
    y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt,
        tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU, f_options=self.f_options)
    mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol,
        rtol=self.rtol, y0=y0, y1=y1)
    accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()
    y_next = y1 if accept_step else y0
    f_next = f1 if accept_step else f0
    t_next = t0 + dt if accept_step else t0
    interp_coeff = _interp_fit_dopri5(y0, y1, k, dt
        ) if accept_step else interp_coeff
    dt_next = _optimal_step_size(dt, mean_sq_error_ratio, safety=self.
        safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5)
    rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next,
        interp_coeff)
    return rk_state
