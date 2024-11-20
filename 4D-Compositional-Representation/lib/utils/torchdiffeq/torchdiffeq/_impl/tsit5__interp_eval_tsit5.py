def _interp_eval_tsit5(t0, t1, k, eval_t):
    dt = t1 - t0
    y0 = tuple(k_[0] for k_ in k)
    interp_coeff = _interp_coeff_tsit5(t0, dt, eval_t)
    y_t = tuple(y0_ + _scaled_dot_product(dt, interp_coeff, k_) for y0_, k_ in
        zip(y0, k))
    return y_t
