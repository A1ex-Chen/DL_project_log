def construct_problem(device, npts=10, ode='constant', reverse=False):
    f = PROBLEMS[ode](device)
    t_points = torch.linspace(1, 8, npts).to(device).requires_grad_(True)
    sol = f.y_exact(t_points)

    def _flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.
            long, device=x.device)
        return x[tuple(indices)]
    if reverse:
        t_points = _flip(t_points, 0).clone().detach()
        sol = _flip(sol, 0).clone().detach()
    return f, sol[0].detach(), t_points, sol
