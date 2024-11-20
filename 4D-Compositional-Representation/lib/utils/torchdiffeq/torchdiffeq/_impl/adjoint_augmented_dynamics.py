def augmented_dynamics(t, y_aug, **f_options):
    y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]
    with torch.set_grad_enabled(True):
        t = t.to(y[0].device).detach().requires_grad_(True)
        y = tuple(y_.detach().requires_grad_(True) for y_ in y)
        func_eval = func(t, y, **f_options)
        vjp_t, *vjp_y_and_params = torch.autograd.grad(func_eval, (t,) + y +
            f_params, tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True,
            retain_graph=True)
    vjp_y = vjp_y_and_params[:n_tensors]
    vjp_params = vjp_y_and_params[n_tensors:]
    vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
    vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for 
        vjp_y_, y_ in zip(vjp_y, y))
    vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
    if len(f_params) == 0:
        vjp_params = torch.tensor(0.0).to(vjp_y[0])
    return *func_eval, *vjp_y, vjp_t, vjp_params
