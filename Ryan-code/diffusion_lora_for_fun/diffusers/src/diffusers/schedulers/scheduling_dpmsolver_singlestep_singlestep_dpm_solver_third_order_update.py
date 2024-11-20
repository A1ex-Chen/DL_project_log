def singlestep_dpm_solver_third_order_update(self, model_output_list: List[
    torch.Tensor], *args, sample: torch.Tensor=None, **kwargs) ->torch.Tensor:
    """
        One step for the third-order singlestep DPMSolver that computes the solution at time `prev_timestep` from the
        time `timestep_list[-3]`.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`):
                The current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    timestep_list = args[0] if len(args) > 0 else kwargs.pop('timestep_list',
        None)
    prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep',
        None)
    if sample is None:
        if len(args) > 2:
            sample = args[2]
        else:
            raise ValueError(' missing`sample` as a required keyward argument')
    if timestep_list is not None:
        deprecate('timestep_list', '1.0.0',
            'Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    sigma_t, sigma_s0, sigma_s1, sigma_s2 = self.sigmas[self.step_index + 1
        ], self.sigmas[self.step_index], self.sigmas[self.step_index - 1
        ], self.sigmas[self.step_index - 2]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)
    m0, m1, m2 = model_output_list[-1], model_output_list[-2
        ], model_output_list[-3]
    h, h_0, h_1 = (lambda_t - lambda_s2, lambda_s0 - lambda_s2, lambda_s1 -
        lambda_s2)
    r0, r1 = h_0 / h, h_1 / h
    D0 = m2
    D1_0, D1_1 = 1.0 / r1 * (m1 - m2), 1.0 / r0 * (m0 - m2)
    D1 = (r0 * D1_0 - r1 * D1_1) / (r0 - r1)
    D2 = 2.0 * (D1_1 - D1_0) / (r0 - r1)
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.solver_type == 'midpoint':
            x_t = sigma_t / sigma_s2 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1_1
        elif self.config.solver_type == 'heun':
            x_t = sigma_t / sigma_s2 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0
                ) * D1 - alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5
                ) * D2
    elif self.config.algorithm_type == 'dpmsolver':
        if self.config.solver_type == 'midpoint':
            x_t = alpha_t / alpha_s2 * sample - sigma_t * (torch.exp(h) - 1.0
                ) * D0 - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1_1
        elif self.config.solver_type == 'heun':
            x_t = alpha_t / alpha_s2 * sample - sigma_t * (torch.exp(h) - 1.0
                ) * D0 - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0
                ) * D1 - sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5
                ) * D2
    return x_t
