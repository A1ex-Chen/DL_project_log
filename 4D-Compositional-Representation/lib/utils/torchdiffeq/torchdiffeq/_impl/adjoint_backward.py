@staticmethod
def backward(ctx, *grad_output):
    t, flat_params, *ans = ctx.saved_tensors
    ans = tuple(ans)
    func, rtol, atol, method, options, f_options = (ctx.func, ctx.rtol, ctx
        .atol, ctx.method, ctx.options, ctx.f_options)
    n_tensors = len(ans)
    f_params = tuple(func.parameters())

    def augmented_dynamics(t, y_aug, **f_options):
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]
        with torch.set_grad_enabled(True):
            t = t.to(y[0].device).detach().requires_grad_(True)
            y = tuple(y_.detach().requires_grad_(True) for y_ in y)
            func_eval = func(t, y, **f_options)
            vjp_t, *vjp_y_and_params = torch.autograd.grad(func_eval, (t,) +
                y + f_params, tuple(-adj_y_ for adj_y_ in adj_y),
                allow_unused=True, retain_graph=True)
        vjp_y = vjp_y_and_params[:n_tensors]
        vjp_params = vjp_y_and_params[n_tensors:]
        vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
        vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for
            vjp_y_, y_ in zip(vjp_y, y))
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
        if len(f_params) == 0:
            vjp_params = torch.tensor(0.0).to(vjp_y[0])
        return *func_eval, *vjp_y, vjp_t, vjp_params
    T = ans[0].shape[0]
    with torch.no_grad():
        adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
        adj_params = torch.zeros_like(flat_params)
        adj_time = torch.tensor(0.0).to(t)
        time_vjps = []
        for i in range(T - 1, 0, -1):
            ans_i = tuple(ans_[i] for ans_ in ans)
            grad_output_i = tuple(grad_output_[i] for grad_output_ in
                grad_output)
            func_i = func(t[i], ans_i, **f_options)
            dLd_cur_t = sum(torch.dot(func_i_.view(-1), grad_output_i_.view
                (-1)).view(1) for func_i_, grad_output_i_ in zip(func_i,
                grad_output_i))
            adj_time = adj_time - dLd_cur_t
            time_vjps.append(dLd_cur_t)
            if len(adj_params) == 0:
                adj_params = torch.tensor(0.0).to(adj_y[0])
            aug_y0 = *ans_i, *adj_y, adj_time, adj_params
            aug_ans = odeint(augmented_dynamics, aug_y0, torch.tensor([t[i],
                t[i - 1]]), rtol=rtol, atol=atol, method=method, options=
                options, f_options=f_options)
            adj_y = aug_ans[n_tensors:2 * n_tensors]
            adj_time = aug_ans[2 * n_tensors]
            adj_params = aug_ans[2 * n_tensors + 1]
            adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for
                adj_y_ in adj_y)
            if len(adj_time) > 0:
                adj_time = adj_time[1]
            if len(adj_params) > 0:
                adj_params = adj_params[1]
            adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_,
                grad_output_ in zip(adj_y, grad_output))
            del aug_y0, aug_ans
        time_vjps.append(adj_time)
        time_vjps = torch.cat(time_vjps[::-1])
        return (*adj_y, None, time_vjps, adj_params, None, None, None, None,
            None)
