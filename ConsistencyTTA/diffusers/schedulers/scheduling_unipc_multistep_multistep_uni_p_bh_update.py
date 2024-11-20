def multistep_uni_p_bh_update(self, model_output: torch.FloatTensor,
    prev_timestep: int, sample: torch.FloatTensor, order: int
    ) ->torch.FloatTensor:
    """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.FloatTensor`):
                direct outputs from learned diffusion model at the current timestep.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            order (`int`): the order of UniP at this step, also the p in UniPC-p.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
    timestep_list = self.timestep_list
    model_output_list = self.model_outputs
    s0, t = self.timestep_list[-1], prev_timestep
    m0 = model_output_list[-1]
    x = sample
    if self.solver_p:
        x_t = self.solver_p.step(model_output, s0, x).prev_sample
        return x_t
    lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
    alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
    sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
    h = lambda_t - lambda_s0
    device = sample.device
    rks = []
    D1s = []
    for i in range(1, order):
        si = timestep_list[-(i + 1)]
        mi = model_output_list[-(i + 1)]
        lambda_si = self.lambda_t[si]
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)
    rks.append(1.0)
    rks = torch.tensor(rks, device=device)
    R = []
    b = []
    hh = -h if self.predict_x0 else h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    factorial_i = 1
    if self.config.solver_type == 'bh1':
        B_h = hh
    elif self.config.solver_type == 'bh2':
        B_h = torch.expm1(hh)
    else:
        raise NotImplementedError()
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    R = torch.stack(R)
    b = torch.tensor(b, device=device)
    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
    else:
        D1s = None
    if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - sigma_t * B_h * pred_res
    x_t = x_t.to(x.dtype)
    return x_t
