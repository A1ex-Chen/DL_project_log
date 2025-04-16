def multistep_uni_c_bh_update(self, this_model_output: torch.FloatTensor,
    this_timestep: int, last_sample: torch.FloatTensor, this_sample: torch.
    FloatTensor, order: int) ->torch.FloatTensor:
    """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.FloatTensor`): the model outputs at `x_t`
            this_timestep (`int`): the current timestep `t`
            last_sample (`torch.FloatTensor`): the generated sample before the last predictor: `x_{t-1}`
            this_sample (`torch.FloatTensor`): the generated sample after the last predictor: `x_{t}`
            order (`int`): the `p` of UniC-p at this step. Note that the effective order of accuracy
                should be order + 1

        Returns:
            `torch.FloatTensor`: the corrected sample tensor at the current timestep.
        """
    timestep_list = self.timestep_list
    model_output_list = self.model_outputs
    s0, t = timestep_list[-1], this_timestep
    m0 = model_output_list[-1]
    x = last_sample
    x_t = this_sample
    model_t = this_model_output
    lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
    alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
    sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]
    h = lambda_t - lambda_s0
    device = this_sample.device
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
    else:
        D1s = None
    if order == 1:
        rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
    else:
        rhos_c = torch.linalg.solve(R, b)
    if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    x_t = x_t.to(x.dtype)
    return x_t
