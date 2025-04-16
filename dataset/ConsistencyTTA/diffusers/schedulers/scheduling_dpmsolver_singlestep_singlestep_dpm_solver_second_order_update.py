def singlestep_dpm_solver_second_order_update(self, model_output_list: List
    [torch.FloatTensor], timestep_list: List[int], prev_timestep: int,
    sample: torch.FloatTensor) ->torch.FloatTensor:
    """
        One step for the second-order singlestep DPM-Solver.

        It computes the solution at time `prev_timestep` from the time `timestep_list[-2]`.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
    t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
    m0, m1 = model_output_list[-1], model_output_list[-2]
    lambda_t, lambda_s0, lambda_s1 = self.lambda_t[t], self.lambda_t[s0
        ], self.lambda_t[s1]
    alpha_t, alpha_s1 = self.alpha_t[t], self.alpha_t[s1]
    sigma_t, sigma_s1 = self.sigma_t[t], self.sigma_t[s1]
    h, h_0 = lambda_t - lambda_s1, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m1, 1.0 / r0 * (m0 - m1)
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.solver_type == 'midpoint':
            x_t = sigma_t / sigma_s1 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        elif self.config.solver_type == 'heun':
            x_t = sigma_t / sigma_s1 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1
    elif self.config.algorithm_type == 'dpmsolver':
        if self.config.solver_type == 'midpoint':
            x_t = alpha_t / alpha_s1 * sample - sigma_t * (torch.exp(h) - 1.0
                ) * D0 - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
        elif self.config.solver_type == 'heun':
            x_t = alpha_t / alpha_s1 * sample - sigma_t * (torch.exp(h) - 1.0
                ) * D0 - sigma_t * ((torch.exp(h) - 1.0) / h - 1.0) * D1
    return x_t
