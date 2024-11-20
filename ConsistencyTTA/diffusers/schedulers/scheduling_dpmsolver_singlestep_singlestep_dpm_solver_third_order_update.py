def singlestep_dpm_solver_third_order_update(self, model_output_list: List[
    torch.FloatTensor], timestep_list: List[int], prev_timestep: int,
    sample: torch.FloatTensor) ->torch.FloatTensor:
    """
        One step for the third-order singlestep DPM-Solver.

        It computes the solution at time `prev_timestep` from the time `timestep_list[-3]`.

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
    t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2
        ], timestep_list[-3]
    m0, m1, m2 = model_output_list[-1], model_output_list[-2
        ], model_output_list[-3]
    lambda_t, lambda_s0, lambda_s1, lambda_s2 = self.lambda_t[t
        ], self.lambda_t[s0], self.lambda_t[s1], self.lambda_t[s2]
    alpha_t, alpha_s2 = self.alpha_t[t], self.alpha_t[s2]
    sigma_t, sigma_s2 = self.sigma_t[t], self.sigma_t[s2]
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
