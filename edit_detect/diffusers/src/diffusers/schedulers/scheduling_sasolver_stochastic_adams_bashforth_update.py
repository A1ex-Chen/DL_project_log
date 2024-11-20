def stochastic_adams_bashforth_update(self, model_output: torch.Tensor, *
    args, sample: torch.Tensor, noise: torch.Tensor, order: int, tau: torch
    .Tensor, **kwargs) ->torch.Tensor:
    """
        One step for the SA-Predictor.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of SA-Predictor at this timestep.

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
    if noise is None:
        if len(args) > 2:
            noise = args[2]
        else:
            raise ValueError(' missing `noise` as a required keyward argument')
    if order is None:
        if len(args) > 3:
            order = args[3]
        else:
            raise ValueError(' missing `order` as a required keyward argument')
    if tau is None:
        if len(args) > 4:
            tau = args[4]
        else:
            raise ValueError(' missing `tau` as a required keyward argument')
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    model_output_list = self.model_outputs
    sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.
        step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    gradient_part = torch.zeros_like(sample)
    h = lambda_t - lambda_s0
    lambda_list = []
    for i in range(order):
        si = self.step_index - i
        alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        lambda_list.append(lambda_si)
    gradient_coefficients = self.get_coefficients_fn(order, lambda_s0,
        lambda_t, lambda_list, tau)
    x = sample
    if self.predict_x0:
        if order == 2:
            temp_sigma = self.sigmas[self.step_index - 1]
            temp_alpha_s, temp_sigma_s = self._sigma_to_alpha_sigma_t(
                temp_sigma)
            temp_lambda_s = torch.log(temp_alpha_s) - torch.log(temp_sigma_s)
            gradient_coefficients[0] += 1.0 * torch.exp((1 + tau ** 2) *
                lambda_t) * (h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.
                exp((1 + tau ** 2) * -h)) / (1 + tau ** 2) ** 2) / (lambda_s0 -
                temp_lambda_s)
            gradient_coefficients[1] -= 1.0 * torch.exp((1 + tau ** 2) *
                lambda_t) * (h ** 2 / 2 - (h * (1 + tau ** 2) - 1 + torch.
                exp((1 + tau ** 2) * -h)) / (1 + tau ** 2) ** 2) / (lambda_s0 -
                temp_lambda_s)
    for i in range(order):
        if self.predict_x0:
            gradient_part += (1 + tau ** 2) * sigma_t * torch.exp(-tau ** 2 *
                lambda_t) * gradient_coefficients[i] * model_output_list[-(
                i + 1)]
        else:
            gradient_part += -(1 + tau ** 2) * alpha_t * gradient_coefficients[
                i] * model_output_list[-(i + 1)]
    if self.predict_x0:
        noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau ** 2 * h)
            ) * noise
    else:
        noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise
    if self.predict_x0:
        x_t = torch.exp(-tau ** 2 * h) * (sigma_t / sigma_s0
            ) * x + gradient_part + noise_part
    else:
        x_t = alpha_t / alpha_s0 * x + gradient_part + noise_part
    x_t = x_t.to(x.dtype)
    return x_t
