def multistep_uni_p_bh_update(self, model_output: torch.Tensor, *args,
    sample: torch.Tensor=None, order: int=None, **kwargs) ->torch.Tensor:
    """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    prev_timestep = args[0] if len(args) > 0 else kwargs.pop('prev_timestep',
        None)
    if sample is None:
        if len(args) > 1:
            sample = args[1]
        else:
            raise ValueError(' missing `sample` as a required keyward argument'
                )
    if order is None:
        if len(args) > 2:
            order = args[2]
        else:
            raise ValueError(' missing `order` as a required keyward argument')
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    model_output_list = self.model_outputs
    s0 = self.timestep_list[-1]
    m0 = model_output_list[-1]
    x = sample
    if self.solver_p:
        x_t = self.solver_p.step(model_output, s0, x).prev_sample
        return x_t
    sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.
        step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0
    device = sample.device
    rks = []
    D1s = []
    for i in range(1, order):
        si = self.step_index - i
        mi = model_output_list[-(i + 1)]
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
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
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x
                .dtype)
    else:
        D1s = None
    if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum('k,bkc...->bc...', rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum('k,bkc...->bc...', rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - sigma_t * B_h * pred_res
    x_t = x_t.to(x.dtype)
    return x_t
