def multistep_dpm_solver_second_order_update(self, model_output_list: List[
    torch.Tensor], sample: torch.Tensor=None, noise: Optional[torch.Tensor]
    =None) ->torch.Tensor:
    """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    sigma_t, sigma_s0, sigma_s1 = self.sigmas[self.step_index + 1
        ], self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    m0, m1 = model_output_list[-1], model_output_list[-2]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, 1.0 / r0 * (m0 - m1)
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.solver_type == 'midpoint':
            x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
        elif self.config.solver_type == 'heun':
            x_t = sigma_t / sigma_s0 * sample - alpha_t * (torch.exp(-h) - 1.0
                ) * D0 + alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0) * D1
    elif self.config.algorithm_type == 'sde-dpmsolver++':
        assert noise is not None
        if self.config.solver_type == 'midpoint':
            x_t = sigma_t / sigma_s0 * torch.exp(-h) * sample + alpha_t * (
                1 - torch.exp(-2.0 * h)) * D0 + 0.5 * (alpha_t * (1 - torch
                .exp(-2.0 * h))) * D1 + sigma_t * torch.sqrt(1.0 - torch.
                exp(-2 * h)) * noise
        elif self.config.solver_type == 'heun':
            x_t = sigma_t / sigma_s0 * torch.exp(-h) * sample + alpha_t * (
                1 - torch.exp(-2.0 * h)) * D0 + alpha_t * ((1.0 - torch.exp
                (-2.0 * h)) / (-2.0 * h) + 1.0) * D1 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2 * h)) * noise
    return x_t
